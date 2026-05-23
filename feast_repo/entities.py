from feast import Entity

customer_entity = Entity(
    name="customer_id",
    description="Customer entity — join key for all customer feature views",
)
