cmake_minimum_required(VERSION 2.6)
PROJECT( ltl2dstar )

OPTION (INTERNAL_BOOST 
  "Use the boost library distributed with ltl2dstar"
  ON)

OPTION (INLINE_BITSETS
  "Inline the bitset operations for performance reasons"
  ON)

SET (UNORDERED_MAP_IMPLEMENTATION "BOOST" CACHE STRING
  "Implementation for hash maps (BOOST,STD,TR1,COMPILER)    ")
SET_PROPERTY(CACHE UNORDERED_MAP_IMPLEMENTATION PROPERTY STRINGS BOOST
  STD TR1 COMPILER)

SET (EXTERNAL_BOOST_INCLUDE CACHE PATH 
  "Specify the location of the boost header files (for external
 boost). Select the directory that has boost/ as subdirectory.")



IF (INTERNAL_BOOST)
 # nothing to do
ELSE(INTERNAL_BOOST)
 INCLUDE_DIRECTORIES( ${EXTERNAL_BOOST_INCLUDE} )
ENDIF(INTERNAL_BOOST)

INCLUDE_DIRECTORIES(".")

IF (INLINE_BITSETS) 
  ADD_DEFINITIONS(-DINLINE_BITSET)
ENDIF(INLINE_BITSETS)


IF (UNORDERED_MAP_IMPLEMENTATION STREQUAL "BOOST")
   ADD_DEFINITIONS(-DHASH_USE_BOOST_UNORDERED_MAP)
ELSEIF (UNORDERED_MAP_IMPLEMENTATION STREQUAL "STD")
   ADD_DEFINITIONS(-DHASH_USE_STD_UNORDERED_MAP)
ELSEIF (UNORDERED_MAP_IMPLEMENTATION STREQUAL "TR1")
   ADD_DEFINITIONS(-DHASH_USE_TR1_UNORDERED_MAP)
ELSEIF (UNORDERED_MAP_IMPLEMENTATION STREQUAL "COMPILER")
   ADD_DEFINITIONS(-DHASH_USE_COMPILER_HASH_MAP)
ELSE ()
   MESSAGE(SEND_ERROR "UNORDERED_MAP_IMPLEMENTATION has to be either
   BOOST, STD or COMPILER")
ENDIF ()

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    IF ("${CMAKE_CXX_FLAGS}" STREQUAL "")
       # Only set flags if there has been no flag set
       SET (CMAKE_CXX_FLAGS "-std=c++11 -Wall" CACHE STRING "Flags used by the compiler during all build types." FORCE)
    ENDIF()
ENDIF()


SET ( ltl2dstar_SRCS 
  common/BitSet.cpp 
  common/RunProgram.win32.cpp 
  common/RunProgram.posix.cpp 
  StutterSensitivenessInformation.cpp 
  LTL2DRA.cpp
  LTL2DSTAR_Scheduler.cpp 
  APMonom.cpp 
  parsers/nba-parser-lbtt.lex.cpp
  parsers/nba-parser-lbtt.tab.cpp
  parsers/nba-parser-promela.lex.cpp
  parsers/nba-parser-promela.tab.cpp
  plugins/PluginManager.cpp
  ltl2dstar_main.cpp
)

ADD_EXECUTABLE ( ltl2dstar ${ltl2dstar_SRCS} )
