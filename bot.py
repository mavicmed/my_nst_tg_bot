import threading
import telebot
from telebot import types
from pathlib import Path
from os import listdir
# from users import UserList
from users import userlist
from logger import logger
from gif import create_gif
from nn_ import ran_nst_
from nn import ran_nst
from gif import create_gif

from queue import Queue

token = open('key.txt', 'r').readline().strip()
imgs = Path('imgs/')

bot = telebot.AsyncTeleBot(token)

q = Queue()

def t_worker():
    """Функция внутри которой обрабатываются tasks на NST из очереди"""
    while True:
        if q.empty():
            continue
        else:
            logger.not_empty_queue(q.qsize())
            try:
                task = q.get()
                task[1](task[0], bot)
                q.task_done()
            except:
                continue

threading.Thread(target=t_worker, ).start()


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    user = userlist.get(message)
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.row('Один', 'Два')
    markup.row('Рестарт')
    msg = """Привет, я умею переносить стиль с одного изображения на другое."""
    msg += "\nБлагодаря курсу от Deep Learning School я могу перенести даже два стиля на одно изображение"
    msg += "\n\nТы хочешь перенести на изображение один стиль или сразу два?"
    bot.reply_to(message, msg, reply_markup=markup)
    bot.register_next_step_handler(message, set_loss_type)
    logger.send_welcome(user)



def general_NST(message):
    user = userlist.get(message)
    print(user)
    msg = 'Blank msg from general_NST'
    if user.loss is None:
        send_welcome(message)
    elif user.loss == 'ONE':
        if user.image_uploaded == 0:
            msg = f'Отправь мне изображение на которое хочешь перенести стиль'
        elif user.image_uploaded == 1:
            msg = f'Отправь мне изображение стиль которого хочешь перенести на предыдущее'
        elif user.image_uploaded > 1:
            msg = f'Я начинаю переносить стиль, это может занять некоторое время'
            q.put((user, ran_nst))
            # t = threading.Thread(target=ran_nst, args=(user, bot))
            # t.start()
            logger.ran_transfer(user)

    elif user.loss == 'TWO':
        if user.image_uploaded == 0:
            msg = f'Отправь мне изображение на которое хочешь перенести стиль'
        elif user.image_uploaded == 1:
            msg = f'Отправь изображение стиль которого будет применен к левой части'
        elif user.image_uploaded == 2:
            msg = f'Отправь изображение стиль которого будет применен к правой части'
        elif user.image_uploaded > 2:
            msg = f'Я начинаю переносить стиль, это может занять некоторое время'
            user.last_message = message
            q.put((user, ran_nst_))
            # t = threading.Thread(target=ran_nst_, args=(user, bot))
            # t.start()
            logger.ran_transfer(user)

    else:
        send_welcome(message)
    bot.reply_to(message, msg)


def set_loss_type(message):
    user = userlist.get(message)
    if message.text == 'Один':
        user.loss = 'ONE'
        logger.set_loss(user, 'ONE')
        print(f'User {user.username} set loss_function_ONE\n')
    elif message.text == 'Два':
        user.loss = 'TWO'
        logger.set_loss(user, 'TWO')
        print(f'User {user.username} set loss_function_TWO\n')
    elif message.text == 'Рестарт':
        user.get_default()
        send_welcome(message)
        logger.restart(user)
    general_NST(message)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    user = userlist.get(message)
    print(f'New img from user {user}')
    if user.image_folder is None:
        usr_folder = Path(f'users/{user.uid}')
        usr_folder.mkdir(parents=True, exist_ok=True)
        img_folder = Path(f'{usr_folder}/{len(listdir(usr_folder))}')
        img_folder.mkdir(parents=True, exist_ok=True)
        user.image_folder = img_folder

    fileID = message.photo[-1].file_id
    file_obj = bot.get_file(fileID).wait()

    file = bot.download_file(file_obj.file_path)
    file = file.wait()
    with open(f'{user.image_folder}/{user.image_uploaded}.jpg', 'wb') as new_file:
        new_file.write(file)
    user.image_uploaded += 1
    logger.new_img(user)
    general_NST(message)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    user = userlist.get(message)
    if message.text == 'Рестарт':
        user.get_default()
        send_welcome(message)
        logger.restart(user)
    else:
        logger.uncnown_message(user, message.text)
        send_welcome(message)


def pooling():
    bot.polling()

if __name__ == '__main__':
    pooling()
