with cleaned as (

    select *
    from {{ ref('int_churn_customers_cleaned') }}

),

final as (

    select
        customer_id,
        churn,

        tenure,
        city_tier,
        warehouse_to_home,
        hour_spend_on_app,
        number_of_device_registered,
        satisfaction_score,
        number_of_address,
        complain,
        order_amount_hike_from_last_year,
        coupon_used,
        order_count,
        day_since_last_order,
        cashback_amount,

        preferred_login_device,
        preferred_payment_mode,
        gender,
        preferred_order_cat,
        marital_status,

        tenure_missing_flag,
        warehouse_to_home_missing_flag,
        hour_spend_on_app_missing_flag,
        day_since_last_order_missing_flag,

        case
            when tenure <= 6 then 1
            else 0
        end as is_new_customer,

        case
            when satisfaction_score <= 2 then 1
            else 0
        end as low_satisfaction_flag,

        case
            when complain = 1 then 1
            else 0
        end as has_complaint,

        case
            when day_since_last_order >= 10 then 1
            else 0
        end as inactive_customer_flag,

        case
            when cashback_amount >= 200 then 1
            else 0
        end as high_cashback_customer_flag,

        batch_id,
        loaded_at,
        loaded_by

    from cleaned

)

select *
from final