with model_input as (

    select *
    from {{ ref('gold_churn_model_input') }}

)

select *
from model_input
where churn = 1