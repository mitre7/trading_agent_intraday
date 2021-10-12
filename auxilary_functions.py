from datetime import datetime
import numpy as np


def returnDataframeBasedOnQuery(df, desired_hour, desired_min, product, date_format):
    """
    This function returns a dataframe based on the input queries of time and product
    :param df: the main dataframe
    :param desired_hour: desired hour
    :param desired_min: desired min
    :param product: desired product
    :param date_format: datatime date format
    :return: A dataframe based on the input queries
    """
    # Dataframe containing only entries of the the given product type
    df_product = df[df['Product'].apply(lambda x: isThisProductType(x, product))]

    # Dataframe containing only entries of the the given product type and given time
    df_ = df_product[
        df_product['Delivery Start'].apply(lambda x: isThisTime(x, desired_hour, desired_min, date_format))]

    return df_


def isThisTime(s, desired_hour, desired_min, date_format):
    """
    This function checks if s has the same time (hour:min) as the input parameters desired_hour/min
    :param s: string variable
    :param desired_hour: desired hour
    :param desired_min: desired minute
    :param date_format: datatime date format
    :return: True if s contains the desired hour and minute
    """
    # Convert string to datetime type
    s_datetime = datetime.strptime(s, date_format)

    # Check if s_datetime has the desired hour and min
    # if s_datetime.hour == desired_hour and s_datetime.minute == desired_min:
    if s_datetime.hour in desired_hour and s_datetime.minute in desired_min:
        return True
    else:
        return False


def isThisProductType(product1, product2):
    """
    Check if the s string is equal to product
    :param product1: product type 1 (hour/half hour/quarter hour)
    :param product2: product type 2 (hour/half hour/quarter hour)
    :return: True, if both input parameters are the same
    """
    return product1 == product2


def convertToDatetype(s, date_format):
    """
    This function converts s string to datetime type
    :param s: string
    :param date_format: datetime
    :return: s_datetime
    """
    s_datetime = datetime.strptime(s, date_format)
    return s_datetime


def getPruductId(s):
    """
    This function generates a product id. For example if delivery starts at 13:15, the product id will be 1315
    :param s: datetime object
    :return: string representing the product id
    """
    return str(s.hour) + ':' + str(s.minute)


def checkIfOrderIdExists(bid_list, order_id):
    """
    Check if bid_id already exists in the bid_list (buy_list or sell_list)
    :param bid_list: a list with all the available buy or sell bids
    :param order_id: the order id that needs to be checked
    :return: the index of order id if it's found, -1 otherwise
    """
    for idx, bid in enumerate(bid_list):
        if bid.id == order_id:
            return idx

    # If id wasn't found, return -1
    return -1


def aggregateBidsFromDifferentProducts(buy_list, sell_list):
    """
    :param buy_list:
    :param sell_list:
    :return:
    """
    buy_list = sorted(buy_list, key=lambda x: x.price, reverse=True)
    sell_list = sorted(sell_list, key=lambda x: x.price, reverse=False)

    cumulative_buy_volume = np.cumsum([buy_obj.volume for buy_obj in buy_list])
    cumulative_sell_volume = np.cumsum([sell_obj.volume for sell_obj in sell_list])

    buy_price = [buy_obj.price for buy_obj in buy_list]
    sell_price = [sell_obj.price for sell_obj in sell_list]

    return cumulative_buy_volume, buy_price, cumulative_sell_volume, sell_price


def returnVolumeAndPrice(lst):
    """
    :param lst: list containing buy or sell objects
    :return: two vectors containing the offered volumes and prices
    """
    volume = [obj.volume for obj in lst]
    price = [obj.price for obj in lst]

    return volume, price


def returnVolume(lst):
    """
    :param lst: list containing buy or sell objects
    :return: vector containing the offered volumes
    """
    volume = [obj.volume for obj in lst]

    return volume


def calcWeightedPrice(lst):
    """
    :param lst: list containing buy or sell objects
    :return: volume-weighted price
    """""
    volume, price = returnVolumeAndPrice(lst)

    weighted_sum = np.dot(volume, price)

    return weighted_sum / np.sum(volume)


def calcPricePercentiles(lst, N, n0, N_max=100):
    """
    :param N_max: distribution percentage considered
    :param lst: list containing buy or sell objects
    :param N: total number of percentiles requested
    :param n0: starting percentile
    :return: a list of the requested percentiles
    """
    price = [obj.price for obj in lst]

    step = int(N_max / N)

    percentiles = [np.percentile(price, n0 + i*step) for i in range(N)]

    return percentiles


def calcVolumePercentiles(lst, N, n0, N_max=100):
    """
    :param N_max: distribution percentage considered
    :param lst: list containing buy or sell objects
    :param N: total number of percentiles requested
    :param n0: starting percentile
    :return: a list of the requested percentiles
    """
    volume = [obj.volume for obj in lst]

    step = int(N_max / N)

    percentiles = [np.percentile(volume, n0 + i*step) for i in range(N)]

    return percentiles


def saveDataFrame(df, path, delivery_hour, delivery_min):
    """
    Save the data frame to the given path
    :param df: given dataframe to be saved
    :param delivery_min: list containing the requested hours
    :param delivery_hour: list containing the requested minutes
    :param path: path to save the data frame
    """
    # path = path + str(self.delivery_hour) + '_' + str(self.delivery_min) + '.csv'
    path = path + '_from_' + str(delivery_hour[0]) + '_' + str(delivery_min[0]) + 'to_' + str(delivery_hour[-1]) + '_' + str(delivery_min[-1]) + '.csv'
    df.to_csv(path, index=False)
