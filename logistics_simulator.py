import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy

import json


PRINT_MODE = 3  # int: [0,4]


def my_softmax(data_vector):
    return [(np.exp(j) / np.sum(np.exp(i) for i in data_vector)) for j in data_vector]


def write_json_data(self, city_labels, location_labels, truck_labels, packages_labels):
    data = {'cities': [], 'location': [], 'trucks': [], 'packages': []}

    for i in range(0, len(city_labels)):
        data['cities'].append({
            'id': city_labels[i].strip(),
            'gps': self.gps_map[city_labels[i].strip()]
        })

    for i in range(0, len(location_labels)):
        data['location'].append({
            'id': location_labels[i].strip(),
            'gps': self.gps_map[location_labels[i].strip()]
        })

    for i in range(0, len(truck_labels)):
        data['trucks'].append({
            'id': truck_labels[i].strip(),
            'gps': self.gps_map[truck_labels[i].strip()]
        })

    for i in range(0, len(packages_labels)):
        data['packages'].append({
            'id': packages_labels[i].strip(),
            'gps': self.gps_map[packages_labels[i].strip()]
        })

    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile, indent=4)


class TruckWorld:

    def __init__(self, truck_labels, packages_labels, location_labels, city_labels, gps_map, noise=.1):

        self.nr_of_trucks = len(truck_labels)
        self.nr_of_packages = len(packages_labels)
        self.nr_of_locations = len(location_labels)
        self.nr_of_cities = len(city_labels)
        self.truck_names = truck_labels
        self.package_names = packages_labels
        self.location_names = location_labels
        self.city_names = city_labels
        self.colors = ['tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7'] * (self.nr_of_trucks // 7 + 1)
        self.noise = noise

        self.trucks = range(self.nr_of_trucks)
        self.packages = range(self.nr_of_packages)
        self.locations = range(self.nr_of_locations)
        self.cities = range(self.nr_of_cities)

        self.actions = [(self.drive_truck, (tr, loc1, loc2, city)) for tr in self.trucks
                        for loc1 in self.locations for loc2 in self.locations for city in self.cities] + \
                       [(self.load_truck, (p, tr, loc)) for p in self.packages for tr in self.trucks for loc in self.locations] + \
                       [(self.unload_truck, (p, tr, loc)) for p in self.packages for tr in self.trucks for loc in self.locations]

        self.action_labels = []

        for a in self.actions:

            if a[0].__name__ == "drive_truck":
                    # and not self.location_names[a[1][1]] == self.location_names[a[1][2]]:
                self.action_labels += [a[0].__name__.upper()+"(" + self.truck_names[a[1][0]]
                                       + ", " + self.location_names[a[1][1]]
                                       + ", " + self.location_names[a[1][2]]
                                       + ", " + self.city_names[a[1][3]] + ")"]

            elif a[0].__name__ == "load_truck":
                self.action_labels += [a[0].__name__.upper()+"(" + self.package_names[a[1][0]]
                                       + ", " + self.truck_names[a[1][1]]
                                       + ", " + self.location_names[a[1][2]] + ")"]

            elif a[0].__name__ == "unload_truck":
                self.action_labels += [a[0].__name__.upper()+"(" + self.package_names[a[1][0]]
                                       + ", " + self.truck_names[a[1][1]]
                                       + ", " + self.location_names[a[1][2]] + ")"]

        self.name = "{}-truck-world noise={}".format(self.nr_of_trucks, self.noise)

        self.gps_map = gps_map

        self.trucks_state = []
        self.location_position = []
        self.packages_position = []

        # Initialize the position of: trucks, packages and locations.
        self.initialize_truck_position()
        self.initialize_location_position()
        self.initialize_package_position()

        # Loaded packages
        self.loaded_packages = np.zeros(len(self.package_names), dtype=int)

        # write_json_data(self, city_labels, location_labels, truck_labels, packages_labels)

    def drive_truck(self, truck, from_loc, to_loc, city):
        """
        ACTION
        Drive the truck from the location "from_loc" to the "to_loc" one. Both locations must be in the same city.
        :param truck: the moved truck
        :param from_loc: the starting location
        :param to_loc: the destination location
        :param city: the locations city
        :return: True if the action is executable and the truck has been moved to the destination location
        """
        preconditions = self.truck_at(truck, from_loc) and self.in_city(from_loc, city) and self.in_city(to_loc, city)
        if preconditions:
            self.trucks_state[truck]['location'] = self.location_names[to_loc]
        return preconditions

    def load_truck(self, package, truck, location):
        """
        ACTION
        Load the package "package" on the truck "truck". Both package and truck must be in the same location.
        :param package: the package to be loaded
        :param truck: the truck to load
        :param location: the location of both package and truck
        :return: True if the action is executable and the package has been loaded into the truck
        """
        preconditions = self.truck_at(truck, location) and self.package_at(package, location)
        if preconditions:
            self.trucks_state[truck]['packages'].append(package)
            self.packages_position[package]['location'] = self.truck_names[truck]
            self.loaded_packages[package] = truck
        return preconditions

    def unload_truck(self, package, truck, location):
        """
        ACTION
        Unload the package "package" from the truck "truck". Both package and truck must be in the same location.
        :param package: the package to be unloaded
        :param truck: the truck to unload
        :param location: the location of both package and truck
        :return: True if the action is executable and the package has been unloaded from the truck
        """
        preconditions = self.truck_at(truck, location) and self.in_vehicle(package, truck)
        if preconditions:
            self.trucks_state[truck]['packages'].remove(package)
            self.packages_position[package]['location'] = self.location_names[location]
            self.loaded_packages[package] = 0
        return preconditions

    def truck_at(self, truck, location):
        """
        PREDICATE
        Check if the truck "truck" is positioned in the location "location"
        :param truck: the truck to check
        :param location: the presumed location of the truck
        :return: True if the truck is in the presumed location
        """
        tp = self.trucks_state[truck]
        if tp['location'].strip() == self.location_names[location].strip():
            return True
        return False

    def package_at(self, package, location):
        """
        PREDICATE
        Check if the package "package" is positioned in the location "location"
        :param package: the package to check
        :param location: the presumed location of the package
        :return: True if the package is in the presumed location
        """
        pp = self.packages_position[package]
        if pp["location"].strip() == self.location_names[location].strip():
            return True
        return False

    def in_city(self, location, city):
        """
        PREDICATE
        Check if the location "location" is placed in the city "city"
        :param location: the location to check
        :param city: the presumed location city
        :return: True if the location is placed in the presumed city
        """
        lp = self.location_position[location]
        if lp.strip() == self.city_names[city].strip():
            return True
        return False

    def in_vehicle(self, package, truck):
        """
        PREDICATE
        Check if the package "package" is loaded in the truck "truck"
        :param package: the package to check
        :param truck: the presumed truck which contains the package
        :return: True if the package is loaded in the presumed truck
        """
        if package in self.trucks_state[truck]["packages"]:
            return True
        return False

    def initialize_truck_position(self):
        """
        Initialize the state of each truck. The state is represented by the city, location and packages of the truck.
        :return:
        """
        for truck in self.truck_names:
            single_truck_state = {}

            for location in self.location_names:

                if self.gps_map[location] == self.gps_map[truck]:
                    single_truck_state['city'] = location.split('_')[0]
                    single_truck_state['location'] = location
                    single_truck_state['packages'] = []
                    self.trucks_state.append(single_truck_state)

    def initialize_location_position(self):
        """
        Initialize the position of each location. The position is defined as the city where the location is placed.
        :return:
        """
        self.location_position = [self.location_names[i].split('_')[0] for i in range(0, len(self.locations))]

    def initialize_package_position(self):
        """
        Initialize the state of each package. The state is represented by the city and location of the package.
        :return:
        """
        for package in self.package_names:
            single_package_state = {}
            matched = False

            for location in self.location_names:

                if not matched and self.gps_map[location] == self.gps_map[package]:
                    single_package_state['city'] = location.split('_')[0]
                    single_package_state['location'] = location
                    self.packages_position.append(single_package_state)
                    matched = True

        assert (len(self.packages_position) == len(self.package_names)), "The initial package gps coordinates must" \
                                                                         " match at least one location gps coordinates."

    def execute_action(self, a):
        """execute action 'a'
        :param a: action to execute
        :return: None if not doable else the new BLOCK_POSITION
        """
        return self.actions[a][0](*self.actions[a][1])

    def sense(self):

        noise_deviation_gps = 1

        # Create the trucks state perception.
        all_perceptions = list()

        for i in range(0, len(self.trucks_state)):

            city_name = self.trucks_state[i]['city'].strip()
            location_name = self.trucks_state[i]['location'].strip()

            # city_name = self.city_names.index(city)
            # location_name = self.location_names.index(location).strip()

            city_gps = self.gps_map[city_name]
            location_gps = self.gps_map[location_name]

            # single_truck_sensor = [self.city_names.index(city), self.location_names.index(location), self.truck_state[i]['pack_distances']]
            single_truck_sensor = location_gps + np.random.normal(0, noise_deviation_gps, 1)
            # np.concatenate((all_trucks_sensor, single_truck_sensor), axis=0)
            all_perceptions.extend(single_truck_sensor)

        # # Create the packages state perception.
        # for package in self.packages_position:
        #     pack_location = package['location'].strip()
        #
        #     # This sensor value should be derived through the RFID triangulation
        #     if pack_location in self.truck_names:
        #         index = self.truck_names.index(pack_location)
        #         location_name = self.trucks_state[index]['location'].strip()
        #         single_pack_sensor = self.gps_map[location_name] + np.random.normal(0, noise_deviation, 1)
        #     else:
        #         single_pack_sensor = self.gps_map[pack_location] + np.random.normal(0, noise_deviation, 1)
        #     all_trucks_sensor.append(single_pack_sensor)



        # Perceive all trucks RFID, one for each package.
        all_trucks_RFID = []
        noise_deviation_RFID = 0.05

        for i in range(len(self.trucks_state)):
            location_name = self.trucks_state[i]['location'].strip()
            location_index = self.location_names.index(location_name)
            single_truck_sensor = []
            for package_index in range(len(self.package_names)):
                try:
                    self.trucks_state[i]['packages'].index(package_index)
                    # single_truck_sensor.append(1 - np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
                    single_truck_sensor.extend(1 - np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
                except ValueError:
                    # single_truck_sensor.append(0 + np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
                    single_truck_sensor.extend(0 + np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
            all_trucks_RFID.extend(single_truck_sensor)

        # Perceive all locations RFID, one for each package.
        all_locations_RFID = []
        for location_index in range(len(self.location_names)):
            single_location_sensor = []
            for package_index in range(len(self.package_names)):
                if self.package_at(package_index, location_index):
                    # single_location_sensor.append(1 - np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
                    single_location_sensor.extend(1 - np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
                else:
                    # single_location_sensor.append(0 + np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
                    single_location_sensor.extend(0 + np.abs(np.random.normal(0, noise_deviation_RFID, 1)))
            all_locations_RFID.extend(single_location_sensor)

        all_RFID_sensors = np.concatenate((all_trucks_RFID, all_locations_RFID), axis=0)
        all_RFID_sensors = [el*10 for el in all_RFID_sensors]
        all_RFID_sensors_scaled = np.zeros(len(all_RFID_sensors))

        all_packages_softmax = []
        package_RFID_count = int(len(all_RFID_sensors)/len(self.package_names))

        for pack in range(len(self.package_names)):
            package_softmax = []
            for i in range(package_RFID_count):
                package_softmax.append(all_RFID_sensors[pack + (i * len(self.package_names))])
            all_packages_softmax.append(my_softmax(package_softmax))

        # all_perceptions.extend(my_softmax(all_RFID_sensors))

        for j in range(package_RFID_count):
            for pack in range(len(self.package_names)):
                all_RFID_sensors_scaled[pack + j*(len(self.package_names))] = all_packages_softmax[pack][j]

        all_perceptions.extend(all_RFID_sensors_scaled)

        # all_perceptions.extend(all_trucks_RFID)
        # all_perceptions.extend(all_locations_RFID)

        # DEBUG
        # print("Perceived sensors information after action execution: {}".format(all_perceptions))

        # DEBUG
        # print(all_trucks_sensor)

        # all_perceptions.extend(self.loaded_packages)

        return all_perceptions

    def complete_walk(self):
        return self._complete_walk([[b] for b in self.trucks])

    def _complete_walk(self,cols):
        result = []
        for c in range(len(cols)):
            for d in range(len(cols)):
                if c != d:
                    result.append(self.actions.index((self.puton, (cols[c][-1], cols[d][-1]))))
                    new_cols = deepcopy(cols)
                    new_cols[d].append(new_cols[c][-1])
                    del new_cols[c]
                    result += self._complete_walk(new_cols)
                    if len(cols[c]) == 1:
                        result.append(self.actions.index((self.ontable, (cols[c][-1],))))
                    else:
                        result.append(self.actions.index((self.puton, (cols[c][-1], cols[c][-2]))))
        return result

    # def show(self,pause=.1):
    #     bp = self.truck_state
    #     plt.clf()
    #     currentAxis = plt.gca()
    #     self.plot_state(currentAxis,bp)
    #     plt.pause(pause)
    #
    # def plot_state(self,currentAxis,bp):
    #     currentAxis.axis([0., self.table_length + 1 / 2, 0., self.nr_of_trucks])
    #     for i in range(self.nr_of_trucks):
    #         currentAxis.add_patch(Rectangle(bp[i], 1, 1,
    #                                         linewidth=1, facecolor=self.colors[i],
    #                                         edgecolor="k"))
    #         plt.text(*(bp[i] + 0.5), self.truck_names[i], ha='center', va='center')



# def g(n,k):
#     return 1 if n==0 else g(n-1,k+1)+(n-1+k)*g(n-1,k)
