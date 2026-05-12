from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import Prefetch, Document, Filter, FieldCondition, MatchAny, MatchValue, FusionQuery
import openai
from langsmith import traceable, get_current_run_tree
from langchain_core.tools import tool
import numpy as np

import psycopg2
from psycopg2.extras import RealDictCursor


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):    
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }

    return response.data[0].embedding


### Item Retrieval Tool

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_items_data(query, k=5):

    qdrant_client = QdrantClient(
        url="http://qdrant:6333"
    )

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[1,1])),
        limit=k
    )

    retrieved_context_ids = []
    retieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retieved_context.append(result.payload["description"])
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_items_context(context):

    formatted_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@tool
def get_formatted_items_context(query: str, top_k: int = 5) -> str:

    """Get the top k context, each representing an inventory item for a given query.
    
    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_items_data(query, top_k)
    formatted_context = process_items_context(context)

    return formatted_context


### Reviews Retrieval Tool

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_prefiltered_reviews_data(query, item_list, k=5):

    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=item_list
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_context_ids = []
    retieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retieved_context.append(result.payload["preprocessed_data"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retieved_context,
        "similarity_scores": similarity_scores
    }


@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, description: {chunk}\n"

    return formatted_context


@tool
def get_formatted_reviews_context(query: str, item_list: list[str], top_k: int = 5) -> str:

    """Get the top k reviews matching a query for a list of prefiltered items.
    
    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered
    
    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    context = retrieve_prefiltered_reviews_data(query, item_list, top_k)
    formatted_context = process_reviews_context(context)

    return formatted_context


### Shopping Cart Agent Tools

### Add to Shopping Cart Tool

def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str: 

    """Add a list of provided items to the shopping cart. 

    Args: 
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity. 
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.

    Returns:
        A list of the items added to the shopping cart.
    """

    conn = psycopg2.connect(
        host="postgres", 
        port=5432, 
        database="tools_database", 
        user="langgraph_user", 
        password="langgraph_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        for item in items:
            product_id = item['product_id']
            quantity = item['quantity']

            qdrant_client = QdrantClient(url="http://qdrant:6333")

            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hybrid-search",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id) 
                                )
                            ]
                        ),              
                        using="text-embedding-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            ).points[0].payload

            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = "USD"

            # Check if item already exist
            check_query = """
                SELECT id, quantity, price
                FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()

            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity

                update_query = """ 
                    UPDATE shopping_carts.shopping_cart_items
                    SET
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """

                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))

            else:
                # Insert new item
                insert_query = """ 
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                    RETURNING id, quantity, price
                """

                cursor.execute(insert_query, (user_id, cart_id, product_id, price, quantity, currency, product_image_url))

    return f"Added {item} to the shopping cart."


### Get Shopping Cart Tool

def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """ 
    Retrieve all items in a user's shopping cart.

    Args:
        user_id: User ID
        cart_id: Cart ID

    Returns:
        List of dictionaries containing cart items
    """

    conn = psycopg2.connect(
        host="postgres", 
        port=5432, 
        database="tools_database", 
        user="langgraph_user", 
        password="langgraph_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        query = """
            SELECT 
                product_id, price, quantity,
                currency, product_image_url,
                (quantity * price) as total_price
            FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s
            ORDER BY added_at DESC
        """
        cursor.execute(query, (user_id, cart_id))

        return [dict(row) for row in cursor.fetchall()]


### Remove from Shopping Cart Tool

def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:
    """ 
    Remove an item completely from the shopping cart.

    Args:
        user_id: User ID
        product_id: Product ID to remove
        cart_id: Cart ID

    Returns:
        True if item was removed, False if item wasn't found.
    """

    conn = psycopg2.connect(
        host="postgres", 
        port=5432, 
        database="tools_database", 
        user="langgraph_user", 
        password="langgraph_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        query = """
            DELETE FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
        """
        cursor.execute(query, (user_id, cart_id, product_id))

        return cursor.rowcount > 0