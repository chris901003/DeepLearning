import os
import time


class Logger:
    def __init__(self, logger_name='root', logger_root='./log_save', save_info=None, email_sender=None, email_key=None):
        """ 對Logger進行初始化
        Args:
             logger_name: 紀錄器名稱，給紀錄器一個名稱比較知道是從哪裡寫出的資料
             logger_root: Logger保存資料根目錄
             save_info: 一個變數名稱後面對應上的是型態，方便保存資料到Logger當中
        """
        self.logger_name = logger_name
        self.time_stamp = time.time()
        self.logger_root = logger_root
        self.email_sender = email_sender
        self.email_key = email_key
        if not os.path.exists(self.logger_root):
            os.mkdir(self.logger_root)
        if save_info is None:
            save_info = dict(default=list())
        self.save_info = save_info

    def append_info(self, var_name, info):
        """ 添加資料到Logger當中
        Args:
             var_name: 保存在save_info當中的key，獲取需要的資料
             info: 要添加上去的資料，如果是特殊形態請自行撰寫添加方式
        """
        target = self.save_info.get(var_name, None)
        if target is None:
            assert ValueError(f'在Logger當中沒有找到{var_name}變數，如果有需要的話請先加入到save_info當中後再調用append_info')
        if isinstance(target, list):
            if isinstance(info, list):
                target.extend(info)
            else:
                target.append(info)
        elif isinstance(target, dict):
            target.update(info)
        else:
            raise NotImplementedError('目前沒有針對該型態實作添加方式，如果有需要請自行添加')
        self.save_info[var_name] = target

    def draw_picture(self, draw_type, **kwargs):
        support_draw_type = {
            'x_y': self.draw_x_y
        }
        draw_function = support_draw_type.get(draw_type, None)
        assert draw_function is not None, f'指定的{draw_type}沒有支援，如果優需要可以自行撰寫，並且放到support_draw_type當中'
        draw_function(**kwargs)

    @staticmethod
    def convert2list(x):
        if not isinstance(x, list):
            x = [x]
        return x

    def draw_x_y(self, save_path, x, y, color=None, title='Graph', x_label='X', y_label='Y',
                 line_style=None, grid=False):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('需安裝matplotlib才可以使用 x_y 函數')
        for idx, x_info in enumerate(x):
            if isinstance(x_info, str):
                info = self.save_info.get(x_info, None)
                assert info is not None, f'在save_info當中沒有找到{x_info}資訊'
                x[idx] = info
        for idx, y_info in enumerate(y):
            if isinstance(y_info, str):
                info = self.save_info.get(y_info, None)
                assert info is not None, f'在save_info當中沒有找到{y_info}資訊'
                y[idx] = info
        if color is None:
            color = (255/255, 100/255, 100/255)
        if line_style is None:
            line_style = '-'
        x = self.convert2list(x)
        y = self.convert2list(y)
        color = self.convert2list(color)
        line_style = self.convert2list(line_style)
        assert len(x) == len(y), f'X與Y的長度不對等，這樣無法畫出圖像 X({len(x)}) Y({len(y)})'
        if len(color) == 1:
            color = color * len(x)
        if len(line_style) == 1:
            line_style = line_style * len(x)
        assert len(x) == len(color) == len(line_style), '關於線條顏色以及線條風格設定數量需要對應'
        for x_axis, y_axis, cur_color, cur_line_style in zip(x, y, color, line_style):
            plt.plot(x_axis, y_axis, cur_line_style, color=cur_color)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(grid)
        save_plt_path = os.path.join(self.logger_root, save_path)
        plt.savefig(save_plt_path)

    def save_to_file(self, save_path, vars_name, separate_type=(' ', '\n'), write_type='w'):
        """ 將指定資料寫入到文件當中
        Args:
            save_path: 保存文件路徑，記得加上副檔名
            vars_name: 需要保存的變數名稱，這裡可以一次保存多個
            separate_type: 一行當中每個變數的分隔方式以及與下個變數的分隔方式
            write_type: 寫入檔案的方式，這裡可以使用['w', 'a']進行寫入
        """
        assert write_type in ['w', 'a'], '寫入格式錯誤'
        separate_type = self.convert2list(separate_type)
        if len(separate_type) == 1:
            separate_type = separate_type * len(vars_name)
        for idx, sep in enumerate(separate_type):
            if isinstance(sep, (tuple, list)):
                pass
            elif isinstance(sep, str):
                sep = (sep, sep)
                separate_type[idx] = sep
            else:
                raise ValueError('separate_type不支援該型態')
        assert len(vars_name) == len(separate_type), '輸出變數長度需要與分離方式相同長度'
        for idx, var_name in enumerate(vars_name):
            if isinstance(var_name, str):
                var = self.save_info.get(var_name, None)
                assert var is not None, f'指定的{var_name}不在save_info當中'
                vars_name[idx] = var
        for idx, var_name in enumerate(vars_name):
            for index, var in enumerate(var_name):
                vars_name[idx][index] = str(vars_name[idx][index])
        save_path = os.path.join(self.logger_root, save_path)
        with open(save_path, write_type) as f:
            for var, sep in zip(vars_name, separate_type):
                res = sep[0].join(var)
                f.write(res)
                f.write(sep[1])

    def send_email(self, subject, send_to, text_info=None, image_info=None):
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        import smtplib
        assert self.email_sender is not None, '需要先指定用哪個帳號進行傳送'
        assert self.email_key is not None, '需要提供傳送電子郵件的密碼'
        assert text_info is not None or image_info is not None, '至少需要選擇文字或是圖像進行傳遞'
        content = MIMEMultipart()
        content['subject'] = subject
        content['from'] = self.email_sender
        content['to'] = send_to
        if text_info is not None:
            if os.path.isfile(text_info):
                with open(text_info, 'r') as f:
                    text_info = f.readlines()
                text_info = ''.join(text_info)
            content.attach(MIMEText(text_info))
        if image_info is not None:
            assert os.path.isfile(image_info), '只支援圖像檔案'
            from email.mime.image import MIMEImage
            from pathlib import Path
            content.attach(MIMEImage(Path(image_info).read_bytes()))
        with smtplib.SMTP(host='smtp.gmail.com', port=587) as smtp:
            try:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self.email_sender, self.email_key)
                smtp.send_message(content)
                print(f'Send email to {send_to}')
            except ValueError:
                print('Email send fail')


def test():
    # 這裡真的會從我的郵件發送出去，所以也不要玩太過分喔
    log = Logger(save_info=dict(x=list(), y=list(), z=list()), email_sender='a0987999103@gmail.com',
                 email_key='lvdcnoxprjblyndt')
    log.append_info('x', [1, 2, 3, 4, 5])
    log.append_info('y', [10, 20, 30, 40, 50])
    log.append_info('z', [1, 15, 28, 38, 55])
    log.draw_x_y(save_path='test.png', x=['x', 'x'], y=['y', 'z'], line_style=['-', '--'], color=[(1, 0, 0), (0, 1, 0)])
    log.save_to_file(save_path='test.txt', vars_name=['x', 'y'])
    text_path = os.path.join(log.logger_root, 'test.txt')
    image_path = os.path.join(log.logger_root, 'test.png')
    log.send_email(subject='Testing', send_to='109project.deeplearning@gmail.com',
                   text_info=text_path, image_info=image_path)


if __name__ == '__main__':
    print('You can try test function')
    print('Hope you can create more useful function')
