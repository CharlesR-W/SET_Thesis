#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:24:08 2023

@author: crw
"""

import pypsa

num_agents = 4

network = pypsa.Network()
generator_voltage_kV = 20.0 # the default network voltage in kV
p_set_kW = 100 # this goes into the generators + loads - I assume it's their set-point somehow?
configuration = "spine"

#default line configuration
add_line = lambda lv_line, lv_bus_1, lv_bus_2: network.add(
        "Line",
        "My line {}".format(lv_line),
        bus0="My bus {}".format(lv_bus_1),
        bus1="My bus {}".format(lv_bus_2),
        x=0.1,
        r=0.01,
)

#default bus configuration
add_bus = lambda lv_bus : network.add(
    "Bus", "My bus {}".format(lv_bus),
    v_nom=generator_voltage_kV
)

#default gen configuration
add_gen = lambda lv_gen, lv_bus, p_set : network.add(
    "Generator",
    "My gen {}".format(lv_gen),
    bus="My bus {}".format(lv_bus),
    p_set=p_set_kW,
    control="PQ"
)

#default load_configuration
add_load = lambda lv_load, lv_bus, p_set: network.add(
    "Load",
    "My load {}".format(lv_load),
    bus="My bus {}".format(lv_bus),
    p_set=p_set_kW
)

if configuration == "spine":
    num_buses = num_agents * 3
    num_lines = 2 + (num_agents-1) * 3

    for lv_bus in range(num_buses):
        add_bus(lv_bus)
    
    #lines numbered left-right, up-down, with the spine last
    # since num_agents = number of 'levels to the spine', loop based on that
    lv_line=0
    for lv_spine in range(num_agents): 
        
        gen_bus = 3*lv_spine
        center_bus = gen_bus + 1
        load_bus = center_bus + 1
        
        next_center_bus = center_bus + 3
        
        add_line(lv_line,gen_bus, center_bus) #gen_bus to center
        lv_line += 1
        
        add_line(lv_line,center_bus,load_bus) #center to load
        lv_line += 1
        
        #don't add the spine bus if this is the last one
        if lv_spine == num_agents - 1:
            pass #passes since later lines add gens + loads
        else:
            add_line(lv_line,center_bus,next_center_bus) #center down
            lv_line += 1
    
        #we'll also add the generators and loads:
        lv_gen = lv_spine
        lv_load = lv_spine
        add_gen(lv_gen, gen_bus, p_set_kW)
        add_load(lv_load, load_bus, p_set_kW)

network.plot()
network.pf()
