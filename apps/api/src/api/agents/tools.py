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



### Item Reviews Retrieval Tool

@traceable(
    name="retrieve_reviews_data",
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
    name="format_retrieved_reviews_context",
    run_type="prompt"
)
def process_reviews_context(context):

    formatted_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, description: {chunk}\n"

    return formatted_context


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

# @traceable(
#     name="add_to_shopping_cart",
#     run_type="tool"
# )
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

# @traceable(
#     name="get_shopping_cart",
#     run_type="tool"
# )
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

# @traceable(
#     name="remove_from_cart",
#     run_type="tool"
# )
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
    


### Warehouse Management Agent Tools

# @traceable(
#     name="check_warehouse_availability",
#     run_type="tool"
# )
def check_warehouse_availability(items: list[dict]) -> dict:

    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.

    Returns:
        A dictionary containing:
        — can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        — warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        — warehouses_partial_fulfillment: list of warehouses with partial availability
        — unavailable_items: list of items that cannot be fulfilled from any warehouse
        — details: detailed breakdown per warehouse with availability for each item
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password"
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": []
            }

            # Check each warehouse for availability
            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM warehouses.inventory
            """
            cursor. execute(warehouse_query)
            warehouses = cursor.fetchall()

            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse["warehouse_id"],
                    "warehouse_name": warehouse["warehouse_name"],
                    "warehouse_location": warehouse["warehouse_location"],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False
                }

                for item in items:
                    product_id = item["product_id"]
                    requested_quantity = item["quantity"]

                    # Check availability in this warehouse
                    availability_query = """
                        SELECT product_id, total_quantity, reserved_quantity, available_quantity
                        FROM warehouses.inventory
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(availability_query, (warehouse["warehouse_id"], product_id))
                    inventory = cursor.fetchone()

                    available_qty = inventory['available_quantity'] if inventory else 0

                    item_detail = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": available_qty > 0 and available_qty < requested_quantity
                    }

                    warehouse_details["items"].append(item_detail)

                    # Track if warehouse can fulfill this item completely
                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False

                    # Track if warehouse has any availability for any item
                    if available_qty > 0:
                        has_any_availability = True

                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append({
                         "warehouse_id": warehouse['warehouse_id'],
                         "warehouse_name": warehouse['warehouse_name'],
                         "warehouse_location": warehouse['warehouse_location'],
                    })
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append({
                         "warehouse_id": warehouse['warehouse_id'],
                         "warehouse_name": warehouse['warehouse_name'],
                         "warehouse_location": warehouse['warehouse_location'],
                    })

                result["details"].append(warehouse_details)

            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item["product_id"]
                requested_quantity = item["quantity"]

                # Get total available quantity across all warehouses
                total_available_query = """
                    SELECT product_id, SUM(available_quantity) as total_available
                    FROM warehouses.inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()

                total_available_qty = total_available['total_available'] if total_available else 0

                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append({
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "total_available_across_warehouses": total_available_qty,
                        "shortage": requested_quantity - total_available_qty
                    })

            result["can_fulfill_completely"] = len(result["warehouses_full_fulfillment"]) > 0 and len(result["unavailable_items"]) == 0

            return result
        
    finally:
        conn.close()


# @traceable(
#     name="reserve_warehouse_items",
#     run_type="tool"
# )
def reserve_warehouse_items(reservations: list[dict]) -> dict:

    """Reserve items from multiple warehouses in a single transaction.

    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                    - warehouse_id: The warehouse to reserve from
                    - product_id: The product to reserve
                    - quantity: The quantity to reserve
    
    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved 
    """

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password"
    )
    conn.autocommit = False     # Use transaction

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "success": False,
                "reserved_items": [],
                "failed_items": []
            }

            for reservation in reservations:
                warehouse_id = reservation['warehouse_id']
                product_id = reservation['product_id']
                quantity = reservation['quantity']

                # Check and lock the inventory row
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location,
                        total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s 
                    FOR UPDATE 
                """  
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()

                if inventory and inventory['available_quantity'] >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s 
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))

                    result["reserved_items"].append({
                        "product_id": product_id,
                        "quantity": quantity,
                        "warehouse_id": warehouse_id,
                        "warehouse_name": inventory['warehouse_name'],
                        "warehouse_location": inventory['warehouse_location']
                    })
                else:
                    result["failed_items"].append({
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "requested": quantity,
                        "available": inventory['available_quantity'] if inventory else 0,
                        "reason": "insufficient_stock" if inventory else "not_in_warehouse"
                    })

            # Only commit if all items were successfully reserved
            if len(result["failed_items"]) == 0:
                conn.commit()
                result["success"] = True
            else:
                conn.rollback()
                result["success"] = False

            return result

    except Exception as e:
        conn.rollback()
        raise e   
    finally:
        conn.close()