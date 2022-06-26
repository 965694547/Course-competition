import datetime
import time


def days_in_month(year, month):
    """
    Inputs:
      year  - an integer between datetime.MINYEAR and datetime.MAXYEAR
              representing the year
      month - an integer between 1 and 12 representing the month

    Returns:
      The number of days in the input month.
    """
    list_normal = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    list_special = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 100 == 0 and year % 400 == 0) or (year % 100 != 0 and year % 4 == 0):
        result = list_special[month - 1]
    else:
        result = list_normal[month - 1]
    print('There are %d days in %d.%d' % (result, year, month))
    return result


def is_valid_date(year, month, day):
    """
    Inputs:
      year  - an integer representing the year
      month - an integer representing the month
      day   - an integer representing the day

    Returns:
      True if year-month-day is a valid date and
      False otherwise
    """
    list_normal = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    list_special = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year not in range(datetime.MINYEAR, datetime.MAXYEAR) or month > 12 \
            or month < 1 or day > 31 or day < 1:
        return False
    else:
        if (year % 100 == 0 and year % 400 == 0) or (year % 100 != 0 and year % 4 == 0):
            result = list_special[month - 1]
        else:
            result = list_normal[month - 1]
        if day <= result:
            print('True')
            return True
        else:
            print('False')
            return False


def days_between(year1, month1, day1, year2, month2, day2):
    """
    Inputs:
      year1  - an integer representing the year of the first date
      month1 - an integer representing the month of the first date
      day1   - an integer representing the day of the first date
      year2  - an integer representing the year of the second date
      month2 - an integer representing the month of the second date
      day2   - an integer representing the day of the second date

    Returns:
      The number of days from the first date to the second date.
      Returns 0 if either date is invalid or the second date is 
      before the first date.
    """
    if is_valid_date(year1, month1, day1) and is_valid_date(year2, month2, day2):
        #year_min = min(year1, year2)
        #year_max = max(year1, year2)
        year_cut = year2 - year1
        year_i = 0
        while not ((year1 + year_i) % 100 == 0 and (year1 + year_i) % 400 == 0) or \
                ((year1 + year_i) % 100 != 0 and (year1 + year_i) % 4 == 0):
            year_i = year_i + 1
        year1 = 2000 + 4 - year_i
        year2 = 2000 + 4 - year_i + year_cut
        date1 = time.strptime('{0}{1}{2}'.format(year1, month1, day1), '%Y%m%d')
        date2 = time.strptime('{0}{1}{2}'.format(year2, month2, day2), '%Y%m%d')
        date1 = datetime.datetime(date1[0], date1[1], date1[2])
        date2 = datetime.datetime(date2[0], date2[1], date2[2])
        day = date2 - date1
        if day.days < 0:
            return 0
        else:
            print('There are %d days between %d.%d.%d and %d.%d.%d' % \
                  (day.days, year1, month1, day1, year1, month2, day2))
            return day.days
    else:
        return 0


def age_in_days(year, month, day):
    """
    Inputs:
      year  - an integer representing the birthday year
      month - an integer representing the birthday month
      day   - an integer representing the birthday day

    Returns:
      The age of a person with the input birthday as of today.
      Returns 0 if the input date is invalid of if the input
      date is in the future.
    """
    year1 = datetime.datetime.now().year
    month1 = datetime.datetime.now().month
    day1 = datetime.datetime.now().day
    todaydate = datetime.date.today()

    def days_between1(year1, month1, day1, year, month, day):
        if is_valid_date(year1, month1, day1) and is_valid_date(year, month, day):
            birthday = time.strptime('{0}{1}{2}'.format(year, month, day), '%Y%m%d')
            date2 = time.strptime('{}'.format(todaydate), '%Y-%m-%d')
            birthday = datetime.datetime(birthday[0], birthday[1], birthday[2])
            date2 = datetime.datetime(date2[0], date2[1], date2[2])
            day = date2 - birthday
            if day.days < 0:
                print('0')
                return 0
            else:
                print('The age of this person are %d days' % (day.days))
                return day.days
        else:
            print('0')
            return 0

    days_between1(year1, month1, day1, year, month, day)


days_in_month(2001, 10)
is_valid_date(2001, 10, 22)
days_between(2001, 10, 22, 2021, 10, 12)
age_in_days(2017, 1, 1)         
