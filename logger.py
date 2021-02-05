from datetime import datetime
from pathlib import Path

log_file = 'var/logs/log.txt'

class Logger():
    def __init__(self, log_path=log_file):
        self.log_path = log_path

    def now(self,):
        return f'{datetime.now():%d.%m.%Y %H:%M:%S}'

    def write_log(mthd):
        def w(self, *args, **kwargs):
            with open(self.log_path, 'a') as log:
                msg = f'{self.now()}  '
                msg += mthd(self, *args, **kwargs) + '\n'
                log.write(msg)
        return w

    @write_log
    def new_user(self, usr):
        """Write into log.txt user data when new user use /start"""
        uid = usr.uid
        username = usr.username
        firstname = usr.firstname

        return f'User entered: {uid} {username} {firstname}'

    @write_log
    def set_loss(self, usr, s):
        """Write loss type into log.txt when user set that"""
        uid = usr.uid
        username = usr.username
        firstname = usr.firstname

        return f'{username} selected {s} loss type'


    @write_log
    def new_img(self, usr):
        msg = f'New image from user {usr.uid} {usr.username} with loss'+\
              f' {usr.loss} save to {usr.image_folder}'
        return msg

    @write_log
    def ran_transfer(self, usr):
        msg = f'Start style transfer for user {usr}.'+\
        f' Result wil be save to {usr.image_folder}'
        return msg

    @write_log
    def sent_result_gif(self, usr):
        msg = f'Result img sent to {usr}. Result folder {usr.image_folder}'
        return msg

    @write_log
    def sent_result_img(self, usr):
        msg = f'Result gif sent to {usr}. Result folder {usr.image_folder}'
        return msg

    @write_log
    def restart(self, usr):
        msg = f'Restarted user data for {usr}'
        return msg

    @write_log
    def send_welcome(self, usr):
        msg = f'Send_welcome to {usr}'
        return msg

    @write_log
    def uncnown_message(self, usr, message):
        msg = f'Unknown message: {message}'+\
        f' From {usr}'
        return msg

    @write_log
    def not_empty_queue(self, q_size):
        msg = f'Queue size: {q_size}'
        return msg

logger = Logger()
