# specify the C++ compiler here
CXX = g++

#DEFINES  =
#DEBUGFLAGS = -g -O1
OPTFLAGS = -O3

CFLAGS = -std=c++11 -I. -Wall -DINLINE_BITSET $(DEFINES) $(DEBUGFLAGS) $(OPTFLAGS)

# uncomment the following line to compile the sample plugins
#PLUGINS = plugins/SamplePlugin.o plugins/DRAReachabilityPlugin.o plugins/WriteDRAPlugin.o

OBJS = $(PLUGINS) common/BitSet.o common/RunProgram.win32.o common/RunProgram.posix.o StutterSensitivenessInformation.o LTL2DRA.o LTL2DSTAR_Scheduler.o APMonom.o parsers/nba-parser-lbtt.lex.o  parsers/nba-parser-lbtt.tab.o parsers/nba-parser-promela.lex.o  parsers/nba-parser-promela.tab.o plugins/PluginManager.o ltl2dstar_main.o

all : ltl2dstar

ltl2dstar : $(OBJS) 
	$(CXX) -g $(LINK_FLAGS) -o ltl2dstar $+

# target for ltl2dstar.exe (compilation with mingw g++ compiler)
ltl2dstar.exe : $(OBJS) 
	$(CXX) -g $(LINK_FLAGS) -o ltl2dstar.exe -static-libgcc -static-libstdc++ $+

%.o : %.cpp
	$(CXX) -c $(CFLAGS) -o $@ $<

clean :
	rm -f ltl2dstar *.o common/*.o parsers/*.o plugins/*.o
