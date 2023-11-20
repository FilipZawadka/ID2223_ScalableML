import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("uni-hopworks-api-key"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login(api_key_value=os.environ['UNI_HOPSWORKS_API_KEY'])
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data().rename(columns={'ph': 'pH'})
    batch_data.columns = batch_data.columns.str.replace('_', ' ', regex=False)
    y_pred = model.predict(batch_data)
    
    dataset_api = project.get_dataset_api()    
   
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 

    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    

    data = {
        'prediction': y_pred,
        'label': list(df.quality),
        'datetime': [now]*len(y_pred),
       }

    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all 3 wine flowers
    print("Number of different flower predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 3:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results)
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 3 different flower predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 3 different wine flower predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

