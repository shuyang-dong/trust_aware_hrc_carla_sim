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


#ifndef LTL2NBA_HPP
#define LTL2NBA_HPP

/** @file
 * Provides wrapper classes for external LTL-to-Buechi translators.
 */

#include "NBA.hpp"
#include "HOA2NBA.hpp"
#include "LTLFormula.hpp"
#include "common/RunProgram.hpp"
#include "common/or_nil.hpp"
#include "parsers/parser_interface.hpp"
#include <cstdio>

/**
 * Virtual base class for wrappers to external LTL-to-Buechi translators.
 */
template <class NBA_t>
class LTL2NBA {
public:
  /** Constructor */
  LTL2NBA() {}
  /** Destructor */
  virtual ~LTL2NBA() {}

  /** Convert an LTL formula to an NBA */
  virtual NBA_t *ltl2nba(LTLFormula& ltl) = 0;
};

/**
 * Wrapper for external LTL-to-Buechi translators using the SPIN interface.
 */
template <class NBA_t>
class LTL2NBA_SPIN : public LTL2NBA<NBA_t> {
public:
  /**
   * Constructor
   * @param path path to the executable
   * @param arguments vector of command line arguments to be passed to the external translator
   */
  LTL2NBA_SPIN(std::string path,
		 std::vector<std::string> arguments=std::vector<std::string>()) :
    _path(path), _arguments(arguments)  {}

  /** Destructor */
  virtual ~LTL2NBA_SPIN() {}

  /** 
   * Convert an LTL formula to an NBA
   * @param ltl
   * @return a pointer to the created NBA (caller gets ownership).
   */
  virtual
  NBA_t *ltl2nba(LTLFormula& ltl) {

    // Create canonical APSet (with 'p0', 'p1', ... as AP)
    LTLFormula_ptr ltl_canonical=ltl.copy();
    APSet_cp canonical_apset(ltl.getAPSet()->createCanonical());
    ltl_canonical->switchAPSet(canonical_apset);

#if (__WIN32__ || _WIN32)
    NamedTempFile spin_outfile;
#else
    AnonymousTempFile spin_outfile;
#endif
    std::vector<std::string> arguments;
    arguments.push_back("-f");
    arguments.push_back(ltl_canonical->toStringInfix());

    arguments.insert(arguments.end(),
		     _arguments.begin(),
		     _arguments.end());
    
    const char *program_path=_path.c_str();
    
    RunProgram spin(program_path,
		    arguments,
		    false,
		    0,
		    &spin_outfile,
		    0);
    
    int rv=spin.waitForTermination();
    if (rv==0) {
      NBA_t *result_nba(new NBA_t(canonical_apset));
      
      FILE *f=spin_outfile.getInFILEStream();
      if (f==NULL) {
	throw Exception("");
      }

      int rc=nba_parser_promela::parse(f, result_nba);
      fclose(f);

      if (rc!=0) {
	throw Exception("Couldn't parse PROMELA file!");
      }
      
      // switch back to original APSet
      result_nba->switchAPSet(ltl.getAPSet());

      return result_nba;
    } else {
      throw Exception("Error running external tool for LTL -> NBA:\n"+spin.getInvocationDetails());
    }
  }

private:
  /** The path */
  std::string _path;

  /** The arguments */
  std::vector<std::string> _arguments;
};


/**
 * Wrapper for external LTL-to-Buechi translators using the LBTT interface.
 */
template <class NBA_t>
class LTL2NBA_LBTT : public LTL2NBA<NBA_t> {
public:
  /**
   * Constructor
   * @param path path to the executable
   * @param arguments vector of command line arguments to be passed to the external translator
   */
  LTL2NBA_LBTT(std::string path,
	       std::vector<std::string> arguments=std::vector<std::string>()) :
    _path(path), _arguments(arguments)  {}

  /** Destructor */
  virtual ~LTL2NBA_LBTT() {}

  /** 
   * Convert an LTL formula to an NBA
   * @param ltl
   * @return a pointer to the created NBA (caller gets ownership).
   */
  virtual NBA_t *ltl2nba(LTLFormula& ltl) {
    // Create canonical APSet (with 'p0', 'p1', ... as AP)
    LTLFormula_ptr ltl_canonical=ltl.copy();
    APSet_cp canonical_apset(ltl.getAPSet()->createCanonical());
    ltl_canonical->switchAPSet(canonical_apset);




    NamedTempFile infile(true);
    NamedTempFile outfile(true);

    std::ostream& o=infile.getOStream();
    o << ltl_canonical->toStringPrefix() << std::endl;
    o.flush();

    std::vector<std::string> arguments(_arguments);
    arguments.push_back(infile.getFileName());
    arguments.push_back(outfile.getFileName());
    
    const char *program_path=_path.c_str();
    
    RunProgram ltl2nba_lbtt(program_path,
			    arguments,
			    false,
			    0,
			    0,
			    0);
    
    int rv=ltl2nba_lbtt.waitForTermination();
    if (rv==0) {
      NBA_t *result_nba=new NBA_t(ltl_canonical->getAPSet());


      FILE *f=outfile.getInFILEStream();
      if (f==NULL) {
	throw Exception("");
      }
      int rc=nba_parser_lbtt::parse(f, result_nba);
      fclose(f);

      if (rc!=0) {
	throw Exception("Couldn't parse LBTT file!");
      }

      // result_nba->print(std::cerr);

      // switch back to original APSet
      result_nba->switchAPSet(ltl.getAPSet());

      return result_nba;
    } else {
      throw Exception("Error running external tool for LTL -> NBA:\n"+ltl2nba_lbtt.getInvocationDetails());
    }
  }

private:
  /** The path */
  std::string _path;

  /** The arguments */
  std::vector<std::string> _arguments;
};





/**
 * Wrapper for external LTL-to-Buechi translators using the generic interface.
 */
template <class NBA_t>
class LTL2NBA_Generic : public LTL2NBA<NBA_t> {
public:
  /**
   * Constructor
   * @param path path to the executable
   * @param arguments vector of command line arguments to be passed to the external translator
   */
  LTL2NBA_Generic(std::string path,
	       std::vector<std::string> arguments=std::vector<std::string>()) :
    _path(path), _arguments(arguments)  {

    std::size_t i = 0;
    while (i < _arguments.size()) {
      std::string& cur = _arguments.at(i);
      if (cur == "<" || cur == ">") {
	if (i+1 < _arguments.size()) {
	  _arguments.at(i+1)=cur + _arguments.at(i+1);
	  _arguments.erase(_arguments.begin() + i);
	  continue;
	} else {
	  // ERROR
	}
      } else if (cur == "") {
	_arguments.erase(_arguments.begin() + i);
	continue;
      } else if (cur.at(0) == '<' ||
		 cur.at(0) == '>') {
	bool ltl = cur.at(0) == '<';
	if (ltl) {
	  if (cur == "<%S" ||
	      cur == "<%L") {
	    if (ltl_spec_index.isNIL()) {
	      ltl_spec_index = i;
	    } else {
	      throw Exception(std::string("ltl2nba-generic: ")
			      +"Multiple LTL formulas specified in parameters: "
			      +_arguments.at(ltl_spec_index)
			      +" and "
			      +cur);
	    }
	  } else {
	      THROW_EXCEPTION(Exception,
			      "Illegal formula specification: "+cur);
	  }
	} else {
	  // automata output spec
	  if (cur == ">%N" ||
	      cur == ">%T" ||
	      cur == ">%H") {
	    if (automaton_spec_index.isNIL()) {
	      automaton_spec_index = i;
	    } else {
	      throw Exception(std::string("ltl2nba-generic: ")+
			      "Multiple automata specified in parameters: "
			      +_arguments.at(automaton_spec_index)
			      +" and "
			      +cur);
	    }
	  } else {
	    throw Exception(std::string("ltl2nba-generic: ")+
			    "Illegal automaton specification: "+cur);
	  }
	}
      } else if (cur.at(0) == '%') {
	if (cur == "%s" || cur == "%S" ||
	    cur == "%l" || cur == "%L") {
	  if (ltl_spec_index.isNIL()) {
	    ltl_spec_index = i;
	  } else {
	    throw Exception(std::string("ltl2nba-generic: ")+
			    "Multiple LTL formulas specified in parameters: "
			    +_arguments.at(ltl_spec_index)
			    +" and "
			    +cur);
	  }
	} else if (cur == "%N" ||
		   cur == "%T" ||
		   cur == "%H") {
	  // automata output spec
	  if (automaton_spec_index.isNIL()) {
	    automaton_spec_index = i;
	  } else {
	    throw Exception(std::string("ltl2nba-generic: ")+
			    "Multiple automata specified in parameters: "
			    +_arguments.at(automaton_spec_index)
			    +" and "
			    +cur);
	  }
	} else {
	  throw Exception(std::string("ltl2nba-generic: ")+
			  "Illegal formula / automaton specification: "+cur);
	}
      }

      // next
      i++;
    }

    if (ltl_spec_index.isNIL()) {
      throw Exception("ltl2nba-generic: No LTL formula specification!");
    }
    if (automaton_spec_index.isNIL()) {
      throw Exception("ltl2nba-generic: No automaton specification!");
    }
  }

  /** Destructor */
  virtual ~LTL2NBA_Generic() {}

  /**
   * Convert an LTL formula to an NBA
   * @param ltl
   * @return a pointer to the created NBA (caller gets ownership).
   */
  virtual NBA_t *ltl2nba(LTLFormula& ltl) {
    // Create canonical APSet (with 'p0', 'p1', ... as AP)
    LTLFormula_ptr ltl_canonical=ltl.copy();
    APSet_cp canonical_apset(ltl.getAPSet()->createCanonical());
    ltl_canonical->switchAPSet(canonical_apset);

    NamedTempFile infile(true);
    NamedTempFile outfile(true);

    TempFile* child_stdin = nullptr;
    TempFile* child_stdout = nullptr;

    std::string ltl_spec = _arguments.at(ltl_spec_index);
    std::string ltl_out;
    if (ltl_spec.at(0) == '<') {
      ltl_spec = ltl_spec.substr(1);
      child_stdin = &infile;
    }
    if (ltl_spec == "%L" || ltl_spec == "%l") {
      ltl_out = ltl_canonical->toStringPrefix();
    } else if (ltl_spec == "%S" || ltl_spec == "%s") {
      ltl_out = ltl_canonical->toStringInfix();
    }
    if (ltl_spec == "%S" || ltl_spec == "%L") {
      std::ostream& o=infile.getOStream();
      o << ltl_out << std::endl;
      o.flush();
    }

    std::string aut_spec = _arguments.at(automaton_spec_index);
    if (aut_spec.at(0) == '>') {
      aut_spec = aut_spec.substr(1);
      child_stdout = &outfile;
    }

    std::vector<std::string> arguments;
    for (std::size_t i=0; i<_arguments.size(); i++) {
      const std::string& cur =_arguments.at(i);
      if (i == ltl_spec_index) {
	if (cur.at(0) == '<') {
	  continue;
	} else if (cur == "%S" || cur == "%L") {
	  arguments.push_back(infile.getFileName());
	} else if (cur == "%s" || cur == "%l") {
	  arguments.push_back(ltl_out);
	} else {
	  throw "Implementation error";
	}
      } else if (i == automaton_spec_index) {
	if (cur.at(0) == '>') {
	  continue;
	} else if (cur =="%N" || cur =="%T" || cur =="%H") {
	  arguments.push_back(outfile.getFileName());
	} else {
	  throw "Implementation error";
	}
      } else {
	arguments.push_back(cur);
      }
    }

    const char *program_path=_path.c_str();

    RunProgram ltl2nba_generic(program_path,
			       arguments,
			       false,
			       child_stdin,
			       child_stdout,
			       0);

    int rv=ltl2nba_generic.waitForTermination();
    if (rv==0) {
      NBA_t *result_nba=new NBA_t(ltl_canonical->getAPSet());

      FILE *f=outfile.getInFILEStream();
      if (f==NULL) {
	throw Exception("");
      }
      int rc;
      if (aut_spec == "%T") {
	rc=nba_parser_lbtt::parse(f, result_nba);
      } else if (aut_spec == "%N") {
	rc=nba_parser_promela::parse(f, result_nba);
      } else if (aut_spec == "%H") {
	std::vector<std::string> hoa_properties;
	rc = HOA2NBA::parse(outfile.getIStream(),
			    result_nba,
			    hoa_properties,
			    canonical_apset) ? 0 : 1;
      } else {
	throw "Implementation error";
      }
      fclose(f);

      if (rc!=0) {
	throw Exception("Couldn't parse automaton file!");
      }

      // result_nba->print_hoa(std::cerr);

      // switch back to original APSet
      result_nba->switchAPSet(ltl.getAPSet());

      return result_nba;
    } else {
      THROW_EXCEPTION(Exception, "Error running external tool for LTL -> NBA:\n"
		      +ltl2nba_generic.getInvocationDetails());
    }
  }

private:
  /** The path */
  std::string _path;

  /** The arguments */
  std::vector<std::string> _arguments;
  or_nil<std::size_t> ltl_spec_index;
  or_nil<std::size_t> automaton_spec_index;
};



#endif
