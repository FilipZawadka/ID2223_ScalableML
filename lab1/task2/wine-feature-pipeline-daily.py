import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("uni-hopworks-api-key"))
   def f():
       g()


def get_random_wine():
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"fixed_acidity" :[random.uniform( 3.8, 15.9)],
                        "volatile_acidity" :[random.uniform( 0.08, 1.58)],
                        "citric_acid" :[random.uniform( 0.0, 1.66)],
                        "residual_sugar" :[random.uniform( 0.6, 65.8)],
                        "chlorides" :[random.uniform( 0.009, 0.611)],
                        "free_sulfur_dioxide" :[random.uniform( 1.0, 289.0)],
                        "total_sulfur_dioxide" :[random.uniform( 6.0, 440.0)],
                        "density" :[random.uniform( 0.98711, 1.03898)],
                        "pH" :[random.uniform( 2.72, 4.01)],
                        "sulphates" :[random.uniform( 0.22, 2.0)],
                        "alcohol" :[random.uniform( 8.0, 14.9)],
                        "white" :[random.choice([True, False])]
                      })
    df['quality'] = random.randint(3, 9)
    return df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(api_key_value=os.environ['UNI_HOPSWORKS_API_KEY'])
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        # stub.deploy("wine_daily")
        with stub.run():
            f()
