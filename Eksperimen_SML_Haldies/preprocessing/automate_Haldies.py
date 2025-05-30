import pandas as pd

def preprocess(df):
    df.drop(columns=["Name", "Cabin", "PassengerId", ], inplace=True)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df = df.copy()
      
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
    
    return df

if __name__ == "__main__":
    df_train = pd.read_csv("../titanic_raw/train.csv")
    df_test = pd.read_csv("../titanic_raw/test.csv")

    preprocessed_train_df = preprocess(df_train)
    preprocessed_test_df = preprocess(df_test)

    preprocessed_train_df.to_csv("titanic_preprocessed_train.csv", index=False)
    preprocessed_test_df.to_csv("titanic_preprocessed_test.csv", index=False)
    print("Preprocessing selesai. Data disimpan sebagai 'titanic_preprocessed_train.csv' dan 'titanic_preprocessed_test.csv'.")