package = "ol-fi-modem"
version = "scm-1"
source = {
   url = "git+https://github.com/aptitudetechnology/ol-fi-modem.git"
}
description = {
   summary = "Software Defined Ol-Fi (SDO) Modem Development Platform - Lua Implementation",
   detailed = [[Implements complete Ol-Fi protocol stack with biomimetic olfactory hardware interface.]],
   homepage = "https://github.com/aptitudetechnology/ol-fi-modem",
   license = "MIT"
}
dependencies = {
   "lua >= 5.1",
   "luasocket",
   "luabitop",
   "luajit-ffi"
}
build = {
   type = "builtin",
   modules = {
      ["ol-fi-modem"] = "ol-fi-modem.lua",
      ["converter"] = "converter.lua"
   }
}
