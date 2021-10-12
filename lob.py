from bid import Bid
from transaction import Transaction
import auxilary_functions as aux
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATE_FORMATS = {'Entry time': '%Y-%m-%dT%H:%M:%S.%fZ',
                'Transaction Time': '%Y-%m-%dT%H:%M:%S.%fZ',
                'Validity time': '%Y-%m-%dT%H:%M:%S.%fZ',
                'Delivery Start': '%Y-%m-%dT%H:%M:%SZ',
                'Delivery End': '%Y-%m-%dT%H:%M:%SZ'}


# This class corresponds to one particular product (hour/half hour/quarter hour)
class LOB:
    # date_format = '%d/%m/%Y %H:%M:%S.%f'
    # date_format = '%Y-%m-%dT%H:%M:%S.%fZ'

    def __init__(self, delivery_day, df):

        # Initialize variables
        self.delivery_day = delivery_day
        self.df = df.copy()
        self.date_formats = DATE_FORMATS
        self.M = len(self.df)

        self.t_threshold = 5  # time period in ms within two transactions of different side are considered the same

        self.current_index = 0  # df index to point at the row until which the market has been reconstructed

        # Get the available products
        self.products = [aux.convertToDatetype(s, self.date_formats['Delivery Start']) for s in df['Delivery Start'].unique()]
        self.product_ids = [aux.getPruductId(s) for s in self.products]

        # dictionaries containing the active buy and sell orders
        self.buy_list = {}
        self.sell_list = {}

        for product_id in self.product_ids:
            self.buy_list[product_id] = []
            self.sell_list[product_id] = []

        # list containing all the transactions
        self.transaction_list = []

        # Just column renaming
        self.df.rename(columns={'Action code': 'Action'},  inplace=True)

        # Convert string-type 'Transaction Time' to datetime type
        self.df['Transaction Time'] = self.df['Transaction Time'].apply(
            lambda x: aux.convertToDatetype(x, self.date_formats['Transaction Time']))

        # Sort data chronologically based on 'Transaction Time' and 'RevisionNo'
        self.df = self.df.sort_values(['Transaction Time', 'RevisionNo'])
        # self.df = temp_sorted

    # def saveDataFrame(self, path):
    #     """
    #     Save the data frame to the given path
    #     :param path: path to save the data frame
    #     """
    #     # path = path + str(self.delivery_hour) + '_' + str(self.delivery_min) + '.csv'
    #     path = path + '_from_' + str(self.delivery_hour[0]) + '_' + str(self.delivery_min[0]) + 'to_' + str(self.delivery_hour[-1]) + '_' + str(self.delivery_min[-1]) + '.csv'
    #     self.df.to_csv(path, index=False)

    # def findActions(self):
    #     """
    #     This function identifies modifications, activations, deactivations and possible cancellations of an order.
    #     The identified action is inputed in the new DataFrame column 'Action'
    #     """
    #
    #     # Series containing the value counts; index = ID, column = number that each ID appeared
    #     counts = self.df['Initial ID'].value_counts()
    #
    #     # Series containing the value counts of IDs that appeared more than once; index = ID, column = number that each ID appeared
    #     same_id_entries = counts[counts > 1]
    #
    #     # Series containing entries that appear only once
    #     unique_id_entries = counts[counts == 1]
    #
    #     # Loop over all same_id_entries
    #     for initial_id in same_id_entries.index:
    #
    #         # Data Frame containing only the entries with Initial ID = initial_id
    #         id_data = self.df[self.df[
    #                               'Initial ID'] == initial_id]  # ATTENTION: It maintains the same indeces as df and it points to the same data frame!!!
    #
    #         M = len(id_data)
    #
    #         # Loop over the entries that have this initial_id
    #         for i in range(1, M):
    #
    #             # It's a modification if bid status Y->Y and there's a cancelling date in the first entry
    #             if pd.notna(id_data.iloc[i - 1]['Cancelling Date']) and id_data.iloc[i]['Is block'] == 'Y' and \
    #                     id_data.iloc[i - 1]['Is block'] == 'Y':
    #
    #                 # Add modification action in the first entry
    #                 epex_index = id_data.iloc[i - 1]['Epex ID']
    #                 self.df.loc[epex_index, 'Action'] = 'modification'
    #
    #             # It's a deactivation if bid status Y->N and there's a cancelling date in the first entry
    #             elif pd.notna(id_data.iloc[i - 1]['Cancelling Date']) and id_data.iloc[i]['Is block'] == 'N' and \
    #                     id_data.iloc[i - 1]['Is block'] == 'Y':
    #
    #                 # Add modification action in the first entry
    #                 epex_index = id_data.iloc[i - 1]['Epex ID']
    #                 self.df.loc[epex_index, 'Action'] = 'deactivation'
    #
    #             # It's a reactivation if bid status N->Y and there's a cancelling date in the first entry
    #             elif pd.notna(id_data.iloc[i - 1]['Cancelling Date']) and id_data.iloc[i]['Is block'] == 'Y' and \
    #                     id_data.iloc[i - 1]['Is block'] == 'N':
    #
    #                 # Add modification action in the first entry
    #                 epex_index = id_data.iloc[i - 1]['Epex ID']
    #                 self.df.loc[epex_index, 'Action'] = 'reactivation'
    #
    #             # Check what's N->N
    #             else:
    #                 # There's no point to check N->N because both orders will be discarded when they are read
    #                 epex_index = id_data.iloc[i - 1]['Epex ID']
    #                 self.df.loc[epex_index, 'Action'] = 'something_else'
    #
    #     # Loop over the unique id entries to track orders with cancellation date
    #     for entry_id in unique_id_entries.index:
    #
    #         # Get the entry with 'Initial ID' = entry_id
    #         entry = self.df[
    #             self.df['Initial ID'] == entry_id]  # ATTENTION:This returns a DataFrame with one entry, not a Series
    #
    #         if pd.notna(entry.iloc[0]['Cancelling Date']) and entry.iloc[0]['Is block'] == 'Y':
    #             epex_index = entry.iloc[0]['Epex ID']
    #             self.df.loc[epex_index, 'Action'] = 'can be cancelled'

    def updateBuyLists(self, row):

        product_id = aux.getPruductId(aux.convertToDatetype(row['Delivery Start'], self.date_formats['Delivery Start']))

        # Check if the order id already exists
        idx = aux.checkIfOrderIdExists(self.buy_list[product_id], row['Order ID'])
        if idx == -1:

            # Create a new Bid object
            new_order = Bid(row['Order ID'], aux.convertToDatetype(row['Delivery Start'], self.date_formats['Delivery Start']),
                            row['Transaction Time'], row['Validity time'],
                            row['Price'], row['Quantity'], row['Side'])
            if row['RevisionNo'] == 1:
                # Add a new list entry
                self.buy_list[product_id].append(new_order)
            else:
                # Display an error message
                # print(f"ERROR: Missing entry with ID: {row['Order ID']} and RevisionNo: {row['RevisionNo']}.")
                pass

        else:

            # Update the existing Bid object
            self.buy_list[product_id][idx].transaction_date = row['Transaction Time']
            self.buy_list[product_id][idx].cancel_date = row['Validity time']
            self.buy_list[product_id][idx].price = row['Price']
            self.buy_list[product_id][idx].volume = row['Quantity']
            if row['Action'] == 'A' or row['Action'] == 'C' or row['Action'] == 'P' or row['Action'] == 'I':
                self.buy_list[product_id][idx].is_activ = True
            elif row['Action'] == 'D' or row['Action'] == 'X' or row['Action'] == 'M':
                self.buy_list[product_id].pop(idx)
            elif row['Action'] == 'H':
                self.buy_list[product_id][idx].is_activ = False
            else:
                print("ERROR: Unknown Action code.")

    def updateSellLists(self, row):

        product_id = aux.getPruductId(aux.convertToDatetype(row['Delivery Start'], self.date_formats['Delivery Start']))

        # Check if the order id already exists
        idx = aux.checkIfOrderIdExists(self.sell_list[product_id], row['Order ID'])
        if idx == -1:

            # Create a new Bid object
            new_order = Bid(row['Order ID'], aux.convertToDatetype(row['Delivery Start'], self.date_formats['Delivery Start']),
                            row['Transaction Time'], row['Validity time'],
                            row['Price'], row['Quantity'], row['Side'])
            if row['RevisionNo'] == 1:
                # Add a new list entry
                self.sell_list[product_id].append(new_order)
            else:
                # Display an error message
                # print(f"ERROR: Missing entry with ID: {row['Order ID']} and RevisionNo: {row['RevisionNo']}.")
                pass

        else:

            # Update the existing Bid object
            self.sell_list[product_id][idx].transaction_date = row['Transaction Time']
            self.sell_list[product_id][idx].cancel_date = row['Validity time']
            self.sell_list[product_id][idx].price = row['Price']
            self.sell_list[product_id][idx].volume = row['Quantity']
            if row['Action'] == 'A' or row['Action'] == 'C' or row['Action'] == 'P' or row['Action'] == 'I':
                self.sell_list[product_id][idx].is_activ = True
            elif row['Action'] == 'D' or row['Action'] == 'X' or row['Action'] == 'M':
                self.sell_list[product_id].pop(idx)
            elif row['Action'] == 'H':
                self.sell_list[product_id][idx].is_activ = False
            else:
                print("ERROR: Unknown Action code.")

    def updateTransactionList(self, row):
        """
        This function is used to keep track of the transactions performed
        :param row: (partially) matched order
        :return: -
        """

        product_id = aux.getPruductId(aux.convertToDatetype(row['Delivery Start'], self.date_formats['Delivery Start']))

        transaction_exists = False

        # Check if a transaction has already been included
        for idx, trans in enumerate(self.transaction_list):

            time_diff = (row['Transaction Time'] - trans.trans_time).microseconds / 1000  # dt in ms

            if time_diff <= self.t_threshold and trans.side != row['Side']:
                if (trans.side == 'Buy' and trans.price >= row['Price']) or (trans.side == 'Sell' and trans.price <= row['Price']):
                    # print(f'Transaction at {trans.trans_time} has been already included!')
                    transaction_exists = True

        # If the transaction has not been included, then append it to the transactions list
        if not transaction_exists:

            # Find the order in order to determine the quantity
            if row['Side'] == 'Buy':
                idx = aux.checkIfOrderIdExists(self.buy_list[product_id], row['Order ID'])
            else:
                idx = aux.checkIfOrderIdExists(self.sell_list[product_id], row['Order ID'])

            if idx == -1:
                # print(f"ERROR: Missing entry with ID: {row['Order ID']}")
                pass
            else:
                if row['Side'] == 'Buy':
                    new_transaction = Transaction(self.buy_list[product_id][idx].price, self.buy_list[product_id][idx].volume - row['Quantity'], row['Transaction Time'], row['Side'])
                else:
                    new_transaction = Transaction(self.sell_list[product_id][idx].price, self.sell_list[product_id][idx].volume - row['Quantity'], row['Transaction Time'], row['Side'])
                self.transaction_list.append(new_transaction)

    def reconstructTillTime(self, desired_date):
        """
        Populate buy_list and sell_list with the bids that are available at the exact desired_date
        :param desired_date: date (datetime type) until reconstruction takes place
        :return: -
        """

        for index, row in self.df.loc[self.current_index:].iterrows():

            if row['Transaction Time'] > desired_date:
                self.current_index = index
                break
            # TODO: Check for expired orders even though it's redundant!
            else:

                # Check if it's a transaction
                if row['Action'] == 'P' or row['Action'] == 'M':
                    self.updateTransactionList(row)

                # Update buy or sell list
                if row['Side'] == 'Buy':
                    self.updateBuyLists(row)
                elif row['Side'] == 'Sell':
                    self.updateSellLists(row)
                else:
                    print("Wrong Side identifier")
                    continue

    def plotCurrentMarketState(self):
        for product_id in self.product_ids:

            # Create the price and quantity lists based on the generated objects
            price_buy = [buy_obj.price for buy_obj in self.buy_list[product_id]]
            price_sell = [sell_obj.price for sell_obj in self.sell_list[product_id]]

            quantity_buy = [buy_obj.volume for buy_obj in self.buy_list[product_id]]
            quantity_sell = [sell_obj.volume for sell_obj in self.sell_list[product_id]]

            # TODO: Invoke mergeOrdersOfSamePrice()

            # Set the font size
            plt.rcParams['font.size'] = '16'

            # Plot the barplots for the available buy and sell bids
            fig, ax = plt.subplots()
            ax.bar(x=price_buy, height=quantity_buy, color='g', label='Buy')
            ax.bar(x=price_sell, height=quantity_sell, color='b', label='Sell')

            # Set figure's details
            # delivery_date = aux.convertToDatetype(self.df.iloc[0]['Delivery Start'], self.date_formats['Delivery Start'])
            fig.suptitle(f'Product: {self.df.iloc[0]["Product"]} starting at {product_id}')

            plt.legend()
            plt.xlabel('Price €/MWh')
            plt.ylabel('Quantity MWh')
            plt.title(f'{self.delivery_day}')

        plt.show()

    def mergeOrdersOfSamePrice(self):
        pass  # TODO

    def calcMarketDepth(self, plot_curves=False):
        cumulative_buy_volume = {}
        buy_price = {}
        cumulative_sell_volume = {}
        sell_price = {}

        for product_id in self.product_ids:
            self.buy_list[product_id] = sorted(self.buy_list[product_id], key=lambda x: x.price, reverse=True)  # Sort buy list so that the most expensive buy bids come first
            self.sell_list[product_id] = sorted(self.sell_list[product_id], key=lambda x: x.price, reverse=False)  # Sort buy list so that the most cheap sell bids come first

            cumulative_buy_volume[product_id] = np.cumsum([buy_obj.volume for buy_obj in self.buy_list[product_id]])
            cumulative_sell_volume[product_id] = np.cumsum([sell_obj.volume for sell_obj in self.sell_list[product_id]])

            buy_price[product_id] = [buy_obj.price for buy_obj in self.buy_list[product_id]]
            sell_price[product_id] = [sell_obj.price for sell_obj in self.sell_list[product_id]]

            if plot_curves:
                fig = plt.figure()
                plt.plot(buy_price[product_id], cumulative_buy_volume[product_id], label='Buy')
                plt.plot(sell_price[product_id], cumulative_sell_volume[product_id], label='Buy')
                plt.title(f'{self.delivery_day}')
                fig.suptitle(f'Product: {self.df.iloc[0]["Product"]} starting at {product_id}')

        if plot_curves:
            plt.show()

        return cumulative_buy_volume, buy_price, cumulative_sell_volume, sell_price

    def mergeAllAvailableProducts(self):
        buy_list = []
        sell_list = []

        for product_id in self.product_ids:
            buy_list = buy_list + self.buy_list[product_id]
            sell_list = sell_list + self.sell_list[product_id]

        return buy_list, sell_list

    def clear_object(self):
        self.current_index = 0

        self.transaction_list = []

        for product_id in self.product_ids:
            self.buy_list[product_id] = []
            self.sell_list[product_id] = []
