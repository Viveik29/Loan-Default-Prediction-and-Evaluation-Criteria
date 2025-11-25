from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",   # handle imbalance
    n_jobs=-1
)

model.fit(X_train, y_train)




if __name__ == '__main__':
    main()
