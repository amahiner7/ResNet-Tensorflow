class UTKFaceData:
    def __init__(self, age, gender, race, date, file_path, train_test_split=None):
        self.age = age
        self.gender = gender
        self.race = race
        self.date = date
        self.file_path = file_path
        self.train_test_split = train_test_split
