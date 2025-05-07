/*
 * This file is part of the program ltl2dstar (http://www.ltl2dstar.de/).
 * Copyright (C) 2005-2018 Joachim Klein <j.klein@ltl2dstar.de>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#define LTL2DSTAR_VERSION "0.5.4"

/** @file
 * Provides main() and command line parsing for ltl2dstar.
 */

#include "LTL2DRA.hpp"
#include "LTL2DSTAR_Scheduler.hpp"
#include "LTL2NBA.hpp"

#include "Configuration.hpp"

#include "LTLFormula.hpp"
#include "LTLPrefixParser.hpp"

#include "DRA2NBA.hpp"

#include "HOA2NBA.hpp"
#include "DRAOptimizations.hpp"

#include "plugins/PluginManager.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cassert>
#include <memory>

#include "common/Exceptions.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>


/**
 * The main class.
 */
class LTL2DSTAR_Main {
public:
  /** Flag: Convert LTL->DRA->NBA? */
  bool flag_dra2nba;

  /** Flag: Print the NBA afert LTL->NBA? */
  bool flag_print_ltl_nba;

  /** Flag: Use limiting with scheduler? */
  bool flag_sched_limits;
  
  /** The limiting factor for the scheduler (alpha) */
  double alpha;
  
  /** Flag: Print statistics for NBA? */
  bool flag_stat_nba;

  /** The input */
  enum {IN_LTL, IN_NBA, IN_COMPLEMENT_NBA} flag_input;

  /** Flag: complement the input */
  bool flag_complement_input;

  /** The output type */
  enum output_type_t {OUT_AUTOMATON, OUT_NBA, OUT_PLUGIN};

  /** Output format type */
  enum output_format_t {OUTFORMAT_NATIVE, OUTFORMAT_HOA, OUTFORMAT_DOT};

  struct output_specification_t {
    output_type_t output_type;
    output_format_t output_format;
    std::string filename;
  };

  /** Vector of output_format_t, filename pairs for extra output */
  std::vector< output_specification_t > outputs;

  /** The main output type */
  output_type_t flag_output_type;

  /** The main output format */
  output_format_t flag_output_format;

  /** The options for Safra's algorithm */
  Options_Safra opt_safra;

  /** The options for LTL2DSTAR */
  LTL2DSTAR_Options opt_ltl2dstar;

  /** Timekeeping: StutterCheck */
  bool stuttercheck_timekeep;
  /** Verbosity: StutterCheck */
  bool stuttercheck_print;

  /** Print usage information or errormessage */
  int usage(char *programname, const std::string& errormsg="") {
    std::cerr << 
      "ltl2dstar v." << LTL2DSTAR_VERSION << "  Copyright (C) 2005-2018 Joachim Klein <j.klein@ltl2dstar.de>\n\n";

    if (errormsg=="") {
      std::cerr <<
        "Usage:" << std::endl <<
        "ltl2dstar [parameters] infile outfile\n" <<
        "\ninfile contains a single line LTL formula in prefix format\n" <<
        "infile/outfile can be '-', meaning stdin/stdout\n\n" <<
        "Parameters: (for details, see doc/ltl2dstar.html documentation)\n"<<
        "===============================================================\n\n" <<

        "External LTL-to-Buechi translator:\n" <<
        "  --ltl2nba=interface:path\n" <<
        "  --ltl2nba=interface:path@parameters\n" <<
        "  --ltl2nba='tool parameters'    (formula placeholders:  %s, %S, %l, %L)\n" <<
        "                                 (automata placeholders: %N, %T, %H)\n" <<
	"                                 (short: -t 'X' => --ltl2nba='X')\n\n" << 

        "Type of generated automata:\n" << 
        "  --automata=rabin,streett\n" <<
        "  --automata=rabin               (default)\n" <<
        "  --automata=streett\n" <<
        "  --automata=original-nba\n\n" <<

        "Input:\n" <<
        "  --input=ltl                    (default)\n" <<
        "  --input=nba                    (NBA in HOA format; short: -B)\n" <<
	"  --complement-input=yes/no      (default=no)\n" <<
	"  --trust-hoa-properties=yes/no  (default=yes)\n\n"

        "Output:\n" <<
        "  --output=automaton             (default)\n" <<
        "  --output=dot   (legacy, equivalent to --output=automaton and\n" <<
        "                  --output-format=dot)\n" <<
        "  --output=nba\n" <<
        "  --output=plugin:NAME\n\n" <<

        "Output format:\n" <<
        "  --output-format=native\n" <<
        "  --output-format=hoa            (short: -H)\n" <<
        "  --output-format=dot            (short: -D)\n\n" <<

	"Extra output (write automaton in additional format):\n"
        "  --extra-output=format:filename (format: native, hoa or dot)\n\n" <<

        "Detailed description of states:\n" <<
        "  --detailed-states=yes/no       (default: no)\n\n" <<

        "Enable/disable \"on-the-fly\" optimizations of Safra's algorithm:\n" <<
        "  --safra=options\n" << 
        "     where options is a comma-seperated list of \n" << 
        "     {all, none, accloop, accsucc, rename, reorder},\n" << 
        "     (a minus '-' in front of an option disables it)\n\n" <<

        "Use bisimulation optimization:\n" <<
        "  --bisimulation=yes/no          (default: yes)\n\n" <<

        "Optimize acceptance conditions:\n" <<
        "  --opt-acceptance=yes/no        (default: yes)\n\n" <<

	"Special handling for deterministic NBA:\n" <<
	"  --dba-direct=yes/no            (default: yes)\n\n" <<

        "Build union automaton for disjunction if beneficial:\n" <<
        "  --union=yes/no                 (default: yes)\n\n" <<

        "Stuttering:\n" << 
        "  --stutter=yes/no               (default: yes)\n" << 
        "  --partial-stutter=yes/no       (default: no)\n\n" << 

        "Use scheck for (co-)safe LTL formulas:\n" <<
        "  --scheck=path\n\n" <<

        "Activate Plugins:\n" << 
        "  --plugin=name:\n" <<
        "  --plugin=name:parameter\n\n" << 

        "Print version and quit:\n" <<
        "  --version\n";

      std::cerr << std::endl; // flush
    } else {
      std::cerr << errormsg << "\n\n" <<
        "Usage:" << std::endl <<
        programname << " [parameters] infile outfile\n" <<
        "\nFor details on the parameters, use\n  " <<
        programname << " --help" << std::endl << std::endl;
    }

    return 1;
  }
  
  /** Get a full line from input stream */
  std::string getLine(std::istream& in) {
    std::stringstream ss;
    in.get(*ss.rdbuf());
    return ss.str();
  }

  /** A data structure for storing the path and the arguments for an external tool */
  typedef std::pair< std::string, std::vector<std::string> > path_argument_pair;
  
  /**
   * Parse the path and store in path_argument_pair.
   * Path has the form 'path' or 'path\@arguments'
   * @param arg_name the name of the argument (for error message)
   * @param path the path argument
   */
  path_argument_pair parse_path(const std::string& arg_name,
                                const std::string& path) {
    size_t pos=path.find('@');
    if (pos!=std::string::npos) {
      if (pos==0) {
        throw CmdLineException("Empty path for parameter '"+arg_name+"'");
      }
      std::string path_=path.substr(0, pos);

      if (pos+1==path.length()) {
        // empty argument part
        return path_argument_pair(path_, 
                                  std::vector<std::string>());
      }
      std::string args_=path.substr(pos+1);

      typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

      boost::char_separator<char> sep(" ");
      tokenizer tokens(args_, sep);

      std::vector<std::string> arg;

      for (tokenizer::iterator it=tokens.begin();
           it!=tokens.end();
           ++it) {
        arg.push_back(*it);
        // std::cerr<< "[" << *it << "]" << std::endl;
      }

      return path_argument_pair(path_,
                                arg);
    } else {
      return path_argument_pair(path, std::vector<std::string>());
    }
  }


  /** 
   * Parse a yes/no value
   * @param arg_name the name of the argument (for error message)
   * @param value the argument value
   * @return true if value=="yes", fals if value=="no"
   */  
  bool parse_yes_no(const std::string& arg_name,
                    const std::string& value) {
    if (value=="yes") {
      return true;
    } else if (value=="no") {
      return false;
    } else {
      throw CmdLineException(std::string("Only yes/no allowed as values for ")+arg_name);
    }
  }

  typedef std::pair<std::string, std::string> string_pair;
  

  /**
   * Split a string at the first occurrence of the character ':'
   */
  string_pair split_at_colon(const std::string& s) {
    size_t pos=s.find(':',0);

    std::string first;
    std::string second;

    if (pos!=std::string::npos) {
      first=s.substr(0, pos);

      if (pos+1!=s.length()) {
        // second not empty
        second=s.substr(pos+1);
      }
    } else {
      first=s;
    }
    
    return string_pair(first, second);
  }


  /** 
   * A value (with optional minus) in a list.
   */
  class ListValue {
  public:
    bool minus;
    std::string value;
  };
  
  /** The type of a vector of ListValue */
  typedef std::vector<ListValue> ListValueVector;

  /**
   * Parse a list of comma-seperated values (optional - in front of values).
   */
  ListValueVector parse_list(const std::string& arg_name,
                             const std::string& value) {
    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

    boost::char_separator<char> sep(",");
    tokenizer tokens(value, sep);

    std::vector<ListValue> list;

    for (tokenizer::iterator it=tokens.begin();
         it!=tokens.end();
         ++it) {

      ListValue lv;
      lv.value=*it;

      if (lv.value.length()==0) {
        throw CmdLineException("Syntax error in "+arg_name);
      }
      
      if (lv.value[0]=='-') {
        lv.minus=true;
        lv.value=lv.value.substr(1);
      } else {
        lv.minus=false;
      }
      list.push_back(lv);
    }

    return list;
  }

  void do_outputs(DRA_ptr dra) {
    NBA_t::shared_ptr nba_from_dra;

    if (dra.get()==0) {
      THROW_EXCEPTION(Exception, "Couldn't generate DRA!");
    }
    if (!dra->isCompact()) {
      dra->makeCompact();
    }

    PluginManager::getManager().afterDRAGeneration(*dra);

    for (output_specification_t& out_spec : outputs) {
      switch (out_spec.output_type) {
      case OUT_NBA:
	if (!nba_from_dra) {
	  nba_from_dra = DRA2NBA::dra2nba<DRA_t, NBA_t>(*dra);
	}
	output(out_spec.filename, nba_from_dra, out_spec.output_format, true);
	break;
      case OUT_AUTOMATON:
	output(out_spec.filename, dra, out_spec.output_format);
	break;
      case OUT_PLUGIN: {
	std::unique_ptr<std::ofstream> out_;

	if (out_spec.filename != "-") {
	  out_.reset(new std::ofstream(out_spec.filename));
	  if (out_->fail()) {
	    THROW_EXCEPTION(Exception, "Couldn't open file "+out_spec.filename+" for output");
	  }
	}
	std::ostream& out = (out_ ? *out_ : std::cout);

	PluginManager::getManager().outputDRA(*dra, out);
	break;
      }
      default:
	THROW_EXCEPTION(Exception, "Implementation error");
      }
    }
  }

  void output(const std::string& filename,
	      NBA_ptr nba,
	      output_format_t output_format,
	      bool was_generated_by_ltl2dstar) {
    std::unique_ptr<std::ofstream> out_;

    if (filename != "-") {
      out_.reset(new std::ofstream(filename));
      if (out_->fail()) {
        THROW_EXCEPTION(Exception, "Couldn't open file "+filename+" for output");
      }
    }
    std::ostream& out = (out_ ? *out_ : std::cout);

    output(out, nba, output_format, was_generated_by_ltl2dstar);
  }

  void output(std::ostream& out,
              NBA_ptr nba,
              output_format_t output_format,
              bool was_generated_by_ltl2dstar) {
    switch (output_format) {
    case OUTFORMAT_NATIVE:
      nba->print_lbtt(out);
      break;
    case OUTFORMAT_HOA:
      nba->print_hoa(out,
                     was_generated_by_ltl2dstar ? "ltl2dstar" : "",
                     was_generated_by_ltl2dstar ? LTL2DSTAR_VERSION : "");
      break;
    case OUTFORMAT_DOT:
      nba->print_dot(out);
      break;
    }
  }

  void output(const std::string& filename, DRA_ptr dra, output_format_t output_format) {
    std::unique_ptr<std::ofstream> out_;

    if (filename != "-") {
      out_.reset(new std::ofstream(filename));
      if (out_->fail()) {
	THROW_EXCEPTION(Exception, "Couldn't open file "+filename+" for output");
      }
    }
    std::ostream& out = (out_ ? *out_ : std::cout);

    output(out, dra, output_format);
  }

  void output(std::ostream& out, DRA_ptr dra, output_format_t output_format) {
    switch (output_format) {
    case OUTFORMAT_NATIVE:
      // standard format
      out << *dra;
      break;
    case OUTFORMAT_HOA:
      // HOA
      dra->print_hoa(out, "ltl2dstar", LTL2DSTAR_VERSION);
      break;
    case OUTFORMAT_DOT:
      dra->print_dot(out);
      break;
    default:
      THROW_EXCEPTION(Exception, "Implementation error");
    }
  }

  /** 
   * Main function. Parse command line and perform actions.
   */
  int main(int argc, char *argv[]) { 
    try {
      std::map<std::string, std::string> defaults;
      defaults["--ltl2nba"]="--ltl2nba=spin:ltl2ba";
      defaults["--automata"]="--automata=rabin";
      defaults["--output"]="--output=automaton";
      defaults["--output-format"]="--output-format=native";
      defaults["--detailed-states"]="--detailed-states=no";
      defaults["--safra"]="--safra=all";
      defaults["--bisimulation"]="--bisimulation=yes";
      defaults["--opt-acceptance"]="--opt-acceptance=yes";
      defaults["--dba-direct"]="--dba-direct=yes";
      defaults["--union"]="--union=yes";
      defaults["--alpha"]="--alpha=10.0";
      defaults["--stutter"]="--stutter=yes";
      defaults["--partial-stutter"]="--partial-stutter=no";
      defaults["--input"]="--input=ltl";
      defaults["--complement-input"]="--complement-input=no";
      defaults["--trust-hoa-properties"]="--trust-hoa-properties=yes";
      //      defaults["--scheck"]=""; // scheck disabled

      // default values...
      flag_dra2nba=false;
      flag_sched_limits=false;

      flag_output_type=OUT_AUTOMATON;
      flag_output_format=OUTFORMAT_NATIVE;
      alpha=1.0;

      stuttercheck_timekeep=false;
      stuttercheck_print=false;

      // options not yet covered
      flag_print_ltl_nba=false;
      flag_stat_nba=false;


      const std::string arg_error("Command line error:\n");

      if (argc>1) {
        if (argv[1]==std::string("--help") ||
            argv[1]==std::string("-h")) {
          return usage(argv[0]);
        }
        if (argv[1]==std::string("--version")) {
          std::cout << "ltl2dstar v." << LTL2DSTAR_VERSION
                    << "\nCopyright (C) 2005-2015 "
                    << "Joachim Klein <j.klein@ltl2dstar.de>\n\n";
          return 0;
        }
      }

      if (argc<=2) {
        usage(argv[0], "No input/output files specified!");
        return 1;
      }


      std::unique_ptr<LTL2NBA<NBA_t> > ltl2nba;

      std::vector<std::string> arguments;
      int argi;
      for (argi=1;argi<argc;argi++) {
        if (strncmp(argv[argi], "--", 2)==0) {
          arguments.push_back(argv[argi]);
        } else if (strcmp(argv[argi], "-t")==0) {
          // handle short-form options
          if (argi+1 < argc) {
            arguments.push_back(std::string("--ltl2nba=")+argv[argi+1]);
            argi++;
          } else {
            throw CmdLineException("Missing value for -t argument");
          }
        } else if (strcmp(argv[argi], "-D")==0) {
            arguments.push_back("--output-format=dot");
        } else if (strcmp(argv[argi], "-H")==0) {
            arguments.push_back("--output-format=hoa");
        } else if (strcmp(argv[argi], "-B")==0) {
            arguments.push_back("--input=nba");
        } else {
          break;
        }
      }

      if (argc - argi > 2) {
        throw CmdLineException("Too many file arguments!");
      }

      if (argc - argi < 2) {
        throw CmdLineException("Too few file arguments!");
      }


      bool default_phase=false;
      std::vector<std::string>::iterator arg_it;
      std::map<std::string, std::string>::iterator default_it
        =defaults.end(); // precaution

      arg_it=arguments.begin();
      while (true) {
        std::string cur_arg;

        // check for end of actual arguments, go to default arguments
        if (!default_phase && arg_it==arguments.end()) {
          default_phase=true;
          default_it=defaults.begin();
        }

        if (arg_it!=arguments.end()) {
          cur_arg=*arg_it;
          ++arg_it;
        } else if (default_phase && 
                   default_it!=defaults.end()) {
          cur_arg=(*default_it).second;
          ++default_it;
        } else {
          // processing finished 
          break;
        }

        // cur_arg is current argument
        std::string::size_type pos;
        pos=cur_arg.find('=', 0); // find =

        if (pos==std::string::npos) {
          throw CmdLineException("No value given for parameter '"+cur_arg+"'");
        }

        std::string arg_name=cur_arg.substr(0, pos);
        std::string arg_value;
        if (cur_arg.length()==pos+1) {
          arg_value="";
        } else {
          arg_value=cur_arg.substr(pos+1);
        }

        // std::cerr << arg_name << " = " << arg_value << std::endl;

        if (!default_phase) {
          // remove default option with same name
          std::map<std::string, std::string>::iterator it;
          it=defaults.find(arg_name);
          if (it!=defaults.end()) {
            defaults.erase(it);
          }
        }


        // parse argument


        if (arg_name=="--ltl2nba") {
          // ---
          // The external LTL-to-Buechi translator
          // ---
          if (arg_value.compare(0, 5, "lbtt:")==0 &&
              arg_value.length()>=6) {
            path_argument_pair pa=parse_path(arg_name, arg_value.substr(5));
            //      std::cerr << pa.first << std::endl;
            ltl2nba.reset(new LTL2NBA_LBTT<NBA_t>(pa.first,
                                                  pa.second));
          } else if (arg_value.compare(0, 5, "spin:")==0 &&
                     arg_value.length()>=6) {
            path_argument_pair pa=parse_path(arg_name, arg_value.substr(5));
            //      std::cerr << pa.first << std::endl;
            ltl2nba.reset(new LTL2NBA_SPIN<NBA_t>(pa.first,
                                                  pa.second));
          } else {
            std::string::size_type pos = arg_value.find_first_of(" :");
            if (pos == std::string::npos || arg_value[pos] == ' ') {
              if (pos != std::string::npos) {
                arg_value[pos] = '@';
              }
              path_argument_pair pa=parse_path(arg_name, arg_value);
              ltl2nba.reset(new LTL2NBA_Generic<NBA_t>(pa.first,
                                                       pa.second));
            } else {
              throw CmdLineException("Illegal argument for "+arg_name
                                     +", unknown call interface '"
                                     +arg_value.substr(0,pos)+"' in "+arg_value);
            }
          }

        } else if (arg_name=="--automata") {
          if (arg_value=="rabin") {
            opt_ltl2dstar.automata=LTL2DSTAR_Options::RABIN;
          } else if (arg_value=="streett") {
            opt_ltl2dstar.automata=LTL2DSTAR_Options::STREETT;
          } else if (arg_value=="rabin,streett" ||
                     arg_value=="streett,rabin") {
            opt_ltl2dstar.automata=LTL2DSTAR_Options::RABIN_AND_STREETT;
          } else if (arg_value=="original-nba") {
            opt_ltl2dstar.automata=LTL2DSTAR_Options::ORIGINAL_NBA;
          } else {
            throw CmdLineException("Illegal value for "+arg_name);
          }

	} else if (arg_name=="--input") {
          if (arg_value=="ltl") {
            flag_input=IN_LTL;
          } else if (arg_value=="nba") {
	    flag_input=IN_NBA;
          } else {
            throw CmdLineException("'"+arg_value+"' is not a valid option for "+arg_name);
	  }

	} else if (arg_name=="--complement-input") {
	  flag_complement_input=parse_yes_no(arg_name, arg_value);

        } else if (arg_name=="--output") {
          if (arg_value=="automaton") {
            flag_output_type=OUT_AUTOMATON;
          } else if (arg_value=="dot") {
            // legacy
            flag_output_type=OUT_AUTOMATON;
            flag_output_format=OUTFORMAT_DOT;
            
            if (!default_phase) {
              // remove default option --output-format
              defaults.erase("--output-format");
            }
          } else if (arg_value=="nba") {
            flag_output_type=OUT_NBA;
          } else if (arg_value.compare(0,7,"plugin:") == 0) {
            flag_output_type=OUT_PLUGIN;
            string_pair sp=split_at_colon(arg_value);
            PluginManager::getManager().setOutputPlugin(sp.second);
            
          } else {
            throw CmdLineException("'"+arg_value+"' is not a valid option for "+arg_name);
          }

        } else if (arg_name=="--output-format") {
          if (arg_value=="native") {
            flag_output_format=OUTFORMAT_NATIVE;
          } else if (arg_value=="hoa") {
            flag_output_format=OUTFORMAT_HOA;
          } else if (arg_value=="dot") {
            flag_output_format=OUTFORMAT_DOT;
          } else {
            throw CmdLineException("'"+arg_value+"' is not a valid option for "+arg_name);
          }

        } else if (arg_name=="--extra-output") {
          string_pair sp=split_at_colon(arg_value);
          output_format_t extra_output_format;

          bool nba = false;
          if (boost::starts_with(sp.first, "nba-")) {
            nba = true;
            sp.first = sp.first.substr(4);
          }
          if (sp.first=="native") {
            extra_output_format=OUTFORMAT_NATIVE;
          } else if (sp.first=="hoa") {
            extra_output_format=OUTFORMAT_HOA;
          } else if (sp.first=="dot") {
            extra_output_format=OUTFORMAT_DOT;
          } else {
            throw CmdLineException(std::string("'")+(nba ? "nba-" : "")+sp.first+"' is not a valid format option for "+arg_name);
          }
          if (sp.second=="") {
            throw CmdLineException("Empty filename for "+arg_name);
          }

          output_specification_t output_spec;
          output_spec.output_type = nba ? OUT_NBA : OUT_AUTOMATON;
          output_spec.output_format = extra_output_format;
          output_spec.filename = sp.second;

          outputs.push_back(output_spec);


        } else if (arg_name=="--verbose") {
          opt_ltl2dstar.verbose_scheduler=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--detailed-states") {
          opt_ltl2dstar.detailed_states=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--trust-hoa-properties") {
          opt_ltl2dstar.trust_hoa_properties=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--time-stuttercheck") {
          stuttercheck_timekeep=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--print-stuttercheck") {
          stuttercheck_print=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--safra") {
          ListValueVector lvv=parse_list(arg_name, arg_value);
          for (ListValueVector::iterator it=lvv.begin();
               it!=lvv.end();
               ++it) {
            ListValue& lv=*it;

            if (lv.value=="all") {
              if (lv.minus) {
                throw CmdLineException("'-all' is illegal for "+arg_name);
              }
              opt_safra.opt_all();
            } else if (lv.value=="none") {
              if (lv.minus) {
                throw CmdLineException("'-none' is illegal for "+arg_name);
              }
              opt_safra.opt_none();
            } else if (lv.value=="accloop") {
              opt_safra.opt_accloop=!lv.minus;
            } else if (lv.value=="accsucc") {
              opt_safra.opt_accsucc=!lv.minus;
            } else if (lv.value=="rename") {
              opt_safra.opt_rename=!lv.minus;
            } else if (lv.value=="reorder") {
              opt_safra.opt_reorder=!lv.minus;
            } else if (lv.value=="nbareject") {
              opt_safra.opt_nba_remove_rejecting=!lv.minus;
            } else {
              throw CmdLineException("'"+lv.value+"' is no valid option for "+arg_name);
            }
          }

        } else if (arg_name=="--stutter") {
          opt_safra.stutter=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--partial-stutter") {
          opt_safra.partial_stutter_check=parse_yes_no(arg_name, arg_value);
        } else if (arg_name=="--stutter-closure") {
          std::cerr << "--stutter-closure is deprecated\n";
          return 1;
        } else if (arg_name=="--bisimulation") {
          opt_ltl2dstar.bisim=parse_yes_no(arg_name, arg_value);

        } else if (arg_name=="--opt-acceptance") {
          opt_ltl2dstar.optimizeAcceptance=parse_yes_no(arg_name, arg_value);

        } else if (arg_name=="--dba-direct") {
          opt_safra.dba_direct=parse_yes_no(arg_name, arg_value);

        } else if (arg_name=="--union") {
          opt_ltl2dstar.allow_union=parse_yes_no(arg_name, arg_value);

        } else if (arg_name=="--scheck") {
          path_argument_pair pa=parse_path(arg_name, arg_value);
        
          if (pa.second.size()!=0) {
            std::cerr << "Warning: Parameters given the scheck tool are currently ignored" << std::endl;
          }
        
          opt_ltl2dstar.scheck_path=pa.first;


        } else if (arg_name=="--plugin") {
          string_pair sp=split_at_colon(arg_value);
          PluginManager::getManager().activatePlugin(sp.first, sp.second);
        } else if (arg_name=="--alpha") {
          if (arg_value=="unlimited") {
            flag_sched_limits=false;
          } else {
            try {
              alpha=boost::lexical_cast<double>(arg_value);
              flag_sched_limits=true;
            } catch (boost::bad_lexical_cast) {
              throw CmdLineException ("Argument to '"+arg_name+"' is not a valid (floating-point) number or 'unlimited'");
            }
          }
      
        } else {
          throw CmdLineException("Unrecognized parameter '"+arg_name+"'");
        }
      }
    
      std::unique_ptr<std::istream> in_;
      std::unique_ptr<std::ostream> out_;

      std::string infilename(argv[argi++]);
      std::string outfilename(argv[argi++]);

      if (infilename != "-") {
        in_.reset(new std::ifstream(infilename));
        if (in_->fail()) {
          THROW_EXCEPTION(Exception, "Couldn't open infile "+infilename);
        }
      }

      std::istream& in = (in_ ? *in_ : std::cin);

      if (opt_ltl2dstar.automata != LTL2DSTAR_Options::ORIGINAL_NBA) {
	output_specification_t output_spec;
	output_spec.output_type = flag_output_type;
	output_spec.output_format = flag_output_format;
	output_spec.filename = outfilename;

	outputs.push_back(output_spec);
      }

      if (flag_input == IN_NBA) {
	NBA_ptr nba(new NBA_t(APSet_cp()));
	std::vector<std::string> hoa_properties;
	if (!HOA2NBA::parse(in, nba.get(), hoa_properties)) {
	  THROW_EXCEPTION(Exception, "Could not read HOA automaton");
	}


	if (opt_safra.opt_nba_remove_rejecting) {
	  NBAAnalysis<NBA_t> nba_analysis(*nba);
	  nba_analysis.removeRejecting();
	}

	if (opt_ltl2dstar.automata==LTL2DSTAR_Options::ORIGINAL_NBA) {
	  output(outfilename, nba, flag_output_format, false);
	  return 0;
	}

	LTL2DRA ltl2dra(opt_safra, ltl2nba.get());
	
	StutterSensitivenessInformation::ptr stutter_info(new StutterSensitivenessInformation());
	if (opt_ltl2dstar.trust_hoa_properties) {
	  stutter_info->checkHOAProperties(hoa_properties);
	}

	DRA_ptr dra = ltl2dra.nba2dra(*nba,
				      0,  // no size limit
				      opt_ltl2dstar.detailed_states,
				      stutter_info);
	if (!dra) {
	  THROW_EXCEPTION(Exception, "Could not construct deterministic automaton!");
	}

	if (opt_ltl2dstar.optimizeAcceptance) {
	  dra->optimizeAcceptanceCondition();
	}

	if (opt_ltl2dstar.bisim) {
	  DRAOptimizations<DRA_t> dra_optimizer;
	  dra=dra_optimizer.optimizeBisimulation(*dra,
						 false,
						 opt_ltl2dstar.detailed_states,
						 false);
	}

	if (flag_complement_input) {
	  dra->considerAsStreett();
	}

	// copy name (if any)
	dra->setName(nba->getName());

	do_outputs(dra);
	return 0;
      }
      
      std::string ltl_string=getLine(in);
      LTLFormula_ptr ltl=LTLPrefixParser::parse(ltl_string);
      if (flag_complement_input) {
	ltl = ltl->negate();
      }
      APSet_cp ap_set=ltl->getAPSet();

      if (stuttercheck_timekeep) {
        StutterSensitivenessInformation::enableTimekeeping();
      }
      if (stuttercheck_print) {
        StutterSensitivenessInformation::enablePrintInfo();
      }


      assert(ltl2nba.get()!=0);
      LTL2DRA ltl2dra(opt_safra, ltl2nba.get());


      if (opt_ltl2dstar.automata==LTL2DSTAR_Options::ORIGINAL_NBA) {
        // We just generate the NBA for the LTL formula
        // and print it
    
        NBA_ptr nba=ltl2dra.ltl2nba(*ltl);
        if (nba.get()==0) {
          THROW_EXCEPTION(Exception, "Can't generate NBA for LTL formula");
        }

	output(outfilename, nba, flag_output_format, false);
        return 0;
      }

      DRA_ptr dra;
      
      LTL2DSTAR_Scheduler ltl2dstar_sched(ltl2dra, 
                                          flag_sched_limits,
                                          alpha);
      
      ltl2dstar_sched.flagStatNBA(flag_stat_nba);
      
      opt_ltl2dstar.opt_safra=opt_safra;
      dra=ltl2dstar_sched.calculate(*ltl, opt_ltl2dstar);

      if (dra) {
	dra->setName(ltl->toStringPrefix());
      }
      do_outputs(dra);
    } catch (CmdLineException& e) {
      return usage(argv[0], 
                   std::string("===================\nCommand line error:\n ")
                   +e.what()+"\n===================\n");
    } catch (Exception& e) {
      e.print(std::cerr);
      return 1;
    } catch (...) {
      std::cerr << "Unknown fatal error..." << std::endl;
      return 1;
    }
    return 0;
  }

};

/**
 * Main function, generate LTL2DSTAR_Main and call main() for that.
 */
int main(int argc, char **argv) {
  LTL2DSTAR_Main ltl2dstar_main;
  return ltl2dstar_main.main(argc, argv);
}
