from logger import logger

class BotUser():
    def __init__(self, message,):
        self.uid = message.from_user.id
        self.username = message.from_user.username
        self.firstname = message.from_user.first_name
        self.menu = 'start'
        self.loss = None
        self.image_uploaded = 0
        self.image_folder = None
        self.last_message = None

    def get_default(self,):
        self.loss = None
        self.image_uploaded = 0
        self.image_folder = None
        self.last_message = None

    def __str__(self,):
        return f'{self.uid} {self.username} {self.firstname}'


class UserList():
    def __init__(self,):
        self.users = {}

    def get(self, message):
        try:
            return self.users[message.from_user.id]
        except:
            self.users[message.from_user.id] = BotUser(message)
            print(f"User entered:", self.users[message.from_user.id])
            logger.new_user(self.users[message.from_user.id])
            return self.users[message.from_user.id]

    def show_users(self,):
        print(self.users)

userlist = UserList()
