from datetime import datetime as dtdt
from datetime import date as dtd
from datetime import timedelta as dttd
from datetime import timezone as tz
from time import perf_counter as pec
from time import perf_counter_ns as nspec # nanoseconds
from time import sleep
import re


def function_timer(func):
    def wrapper(*args, **kwargs):
        print(func)
        t_start = nspec()
        out = func(*args, **kwargs)
        t_stop = nspec()
        print(f"{(t_stop-t_start)/1e6:.3f} ms")
        return out
    return wrapper

class Stopwatch(): # nanoseconds
    total = lap = 0
    def __init__(self):
        self.now = self.start = nspec() # ns integer
    def reset(self): self.now = self.starttime = nspec()
    def check(self): return nspec()-self.now
    def __call__(self):
        now = nspec()
        self.total = now-self.start
        self.lap = now-self.now
        self.now = now

def ticks(seconds=10, rate=1):
    ns_rate = int(rate*1e9)
    f_acc = str(seconds)
    if "." in f_acc: f_acc = len(f_acc[f_acc.index(".")+1:])+1
    else: f_acc = 1
    sw = stopwatch()
    while 0<seconds:
        sw()
        yield round(seconds,f_acc) if f_acc>1 else round(seconds)
        t_delta = (ns_rate-sw.lap)/1e9
        if t_delta>0: sleep(t_delta)
        sw()
        seconds -= sw.lap/1e9


#
def format_datetime(date):
    if type(date)==dtdt: return date
    return dtdt.fromisoformat(date)
def format_iso(date):
    if type(date)==str: return date
    return date.isoformat()
def format_clean(date, ms=False):
    if date.microsecond: return date.isoformat()[:-6-7*(1-ms)]
    return date.isoformat()[:-6]
def addtime(date, years=0, months=0, weeks=0, days=0,
           hours=0, minutes=0, seconds=0, microseconds=0):
    iso = type(date)==str
    if not iso: date = date.isoformat()
    a, m, date = date.split("-", 2)
    m = int(m)+months
    intm = m//13
    date = str(min(max(int(a)+years+intm, 1), 9999)).rjust(4, "0")+"-"+str(max(m-intm*12, 1)).rjust(2, "0")+"-"+date
    date = format_datetime(date)+dttd(days=days+weeks*7, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)
    if iso: return date.isoformat()
    return date
#


# return integers
def duration(date1, date2=None):
    if date2 is None: date2 = today()
    return format_datetime(date2)-format_datetime(date1)

DURATION_FORMAT_ZEROS = [0]*6
def duration_format(date1, date2=None,
            years=False, days=False, hours=False,
            minutes=False, seconds=False, microseconds=False):
    dur = duration(date1, date2)
    b = [years, days, hours, minutes, seconds, microseconds]
    if sum(b)==0: return dur
    d = DURATION_FORMAT_ZEROS.copy()
    d[1] = dur.days
    d[4] = dur.seconds
    if microseconds: d[5] = dur.microseconds
    if years: # 365 days
        d[0] = int(d[1]/365)
        d[1] = d[1]%365
    if hours: # 3600 seconds
        d[2] = int(d[4]/3600)
        d[4] = d[4]%3600
    if minutes: # 60 seconds
        d[3] = int(d[4]/60)
        d[4] = d[4]%60
    if not days:
        if hours: d[2] += d[1]*24
        elif minutes: d[3] += d[1]*1440
        elif seconds: d[4] += d[1]*86400
        elif microseconds: d[5] += d[1]*86400000000
        d[1] = 0
    if not seconds:
        if microseconds: d[5] += d[4]*1000000
        d[4] = 0
    return [x for i,x in enumerate(d) if b[i]]
#




# return datetime
def now(): return dtdt.now(tz.utc)
def today(): return dtdt.fromisoformat(dtd.today().isoformat())
def tomorrow(): return addtime(dtd.today(), days=1)
def future(*args, **kwargs):
    datetime_now = now()
    kwargs["microseconds"] = -datetime_now.microsecond
    return addtime(datetime_now, *args, **kwargs)

def next_time(hour, minute=0, second=0):
    date = now()
    hours = (hour%24-date.hour)
    mins = (minute%60-date.minute)
    secs = (second%60-date.second)
    over = max(-(hours*3600+mins*60+secs), 0)
    if over: over = over//(24*3600)+1
    return addtime(date, days=over, hours=hours, minutes=mins, seconds=secs, microseconds=-date.microsecond)
#

# string formats
def clean(date=None, ms=False):
    if date is None: date = now()
    if date.microsecond: return date.isoformat()[:-6-7*(1-ms)]
    return date.isoformat()[:-6]

def cleandate(): return clean().rsplit("T", 1)[0]

def cleantuple(*args, **kwargs):
    return clean(*args, **kwargs).split("T")

def print_prefix(date=None, depth=0):
    return " ".join(cleantuple(date))+" |"+" "*int(depth)

PATHTIME_STRFTIME = "%Y-%m-%d_%Hh%Mm%Ss"
PATHTIME_RE = re.compile(r'(?:(\d+)\-)(?:(\d+)\-)(?:(\d+))_(?:(\d+)h)(?:(\d+)m)(?:(\d+)s)')
PATHTIME_RE_OLD = re.compile(r'(?:(\d+)a)(?:(\d+)m)(?:(\d+)d)_(?:(\d+)h)(?:(\d+)m)(?:(\d+)s)')
def pathtime(date): # turn datetime object to a path friendly string or reverse it
    if type(date)==dtdt: # write
        return date.strftime(PATHTIME_STRFTIME)
    if type(date)==str and ((m:=PATHTIME_RE.match(date)) or (m:=PATHTIME_RE_OLD.match(date))): # read
        return dtdt(tzinfo=tz.utc, year=int(m.group(1)),
                    month=int(m.group(2)), day=int(m.group(3)),
                    hour=int(m.group(4)), minute=int(m.group(5)),
                    second=int(m.group(6)))
#





if __name__ == "__main__":
##    x = now()
##    print(x.year, type(x.year))
    
##    print(cleandate())
##    print(pathtime("2024a01m29d_20h04m15s"))
    
####    t = today()
######    t = tomorrow()
######    dur = duration_format(future(1), future(2, 6), years=True, days=True)
####
######    t = future(10, months=1)
####    print(t)
####    t = pathtime(t)
####    print(t)
####    print(pathtime(t))

##    print(next_time(1, 30, 60))
##    print(next_time(10, 30, 0))
    
##    t = utcadd(t, days=1)
##    print(t)
    
##    print(today_iso())
##    print(days_since("2013-01-13"))
    pass
