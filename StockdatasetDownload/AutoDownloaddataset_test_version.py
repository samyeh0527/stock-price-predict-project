# -*- coding: UTF-8 -*-

import os
import sys
import time

from random import randint

from selenium import webdriver


driver = webdriver.Chrome()
# go to the Yahoo Finance page
driver.get("https://finance.yahoo.com/")
time.sleep(10)

element = driver.find_element_by_id("yfin-usr-qry")

element.send_keys("2330.tw")

button = driver.find_element_by_id("header-desktop-search-button")

button.click()

