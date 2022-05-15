import xlwings as xw
import psutil
import os


def kill_excel_by_pid():
    # 先清理一下可能存在的Excel进程
    pids = psutil.pids()
    for pid in pids:
        try:
            p = psutil.Process(pid)
            # print('pid=%s,pname=%s' % (pid, p.name()))
            # 关闭excel进程
            if p.name() == 'EXCEL.EXE':
                cmd = 'taskkill /F /IM EXCEL.EXE'
                os.system(cmd)
        except Exception as e:
            print(e)


def refresh_ptable():
    App = xw.App(visible=False, add_book=False)
    wb = App.books.open('数据透视表.xlsx')
    sheet = wb.sheets('数据透视表')
    print('sheet:', sheet)
    # 进行透视表的刷新
    # res = sheet.api.PivotTables("数据透视表1").PivotCache().Refresh()
    res = sheet.api.PivotTables(1).PivotCache().Refresh()
    print(res)

    # 保存刷新的结果
    wb.save()
    wb.close()
    App.quit()


kill_excel_by_pid()
refresh_ptable()
kill_excel_by_pid()
