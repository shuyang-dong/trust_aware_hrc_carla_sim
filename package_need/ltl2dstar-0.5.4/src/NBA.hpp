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


#ifndef NBA_H
#define NBA_H

/** @file
 * Provides class NBA to store a nondeterministic Büchi automaton.
 */

#include "common/Exceptions.hpp"
#include "common/Index.hpp"
#include "common/BitSet.hpp"
#include "common/BitSetIterator.hpp"

#include <boost/iterator/iterator_facade.hpp>
#include <boost/shared_ptr.hpp>

#include <string>
#include <iostream>
#include <iterator>
#include <vector>
#include <map>
#include <utility>

#include "NBA_I.hpp"
#include "APSet.hpp"
#include "GraphAlgorithms.hpp"
#include "StutterSensitivenessInformation.hpp"

#include "cpphoafparser/parser/hoa_parser_helper.hh"

// forward declaration of NBA_State
template <typename Label, template <typename N> class EdgeContainer > class NBA_State;


/**
 * A nondeterministic Büchi automaton.
 * See class DA for description of template parameters.
 */
template <typename Label, template <typename N> class EdgeContainer >
class NBA : public NBA_I {
public:
  NBA(APSet_cp apset);
  virtual ~NBA();

  /** The type of the states of the NBA. */  
  typedef NBA_State<Label,EdgeContainer> state_type;

  /** The type of the graph (ie the NBA class itself). */
  typedef NBA<Label,EdgeContainer> graph_type;

  /** The type of an iterator over the edges of a state. */
  typedef typename EdgeContainer<BitSet>::iterator edge_iterator;

  /** 
   * The type of an edge, consisting of the label and a pointer to a BitSet
   * of the target states.
   */
  typedef std::pair<Label, BitSet*> edge_type;

  state_type* newState();

  /** Get number of states. */
  std::size_t size() {return _index.size();}

  /** Type of an iterator over the states (by reference) */
  typedef typename Index<state_type>::ref_iterator iterator;
  
  /** An iterator over the states (by reference) pointing to the first state. */
  iterator begin() {return _index.begin_ref();}

  /** An iterator over the states (by reference) pointing to the first state. */
  iterator end() {return _index.end_ref();}

  /** Array index operator, get the state with index i. */
  state_type* operator[](std::size_t i) {
    return _index.get(i);
  }

  /** Get the size of the underlying APSet. */
  unsigned int getAPSize() const {return _apset->size();};

  /** Get a const reference to the underlying APSet. */
  const APSet& getAPSet() const {return *_apset;};

  /** Get a const pointer to the underlying APSet. */
  APSet_cp getAPSet_cp() const {return _apset;};

  /** Switch the APSet to another with the same number of APs. */
  void switchAPSet(APSet_cp new_apset) {
    if (new_apset->size()!=_apset->size()) {
      THROW_EXCEPTION(IllegalArgumentException, "New APSet has to have the same size as the old APSet: " +
		      boost::lexical_cast<std::string>(*_apset)+" -> "+
		      boost::lexical_cast<std::string>(*new_apset));
    }

    _apset=new_apset;
  }


  /** Get the index for a state. */
  std::size_t getIndexForState(const state_type *state) const {
    return _index.get_index(state);
  }

  /** Set the start state. */
  void setStartState(state_type *state) {_start_state=state;};

  /**
   * Get the start state.
   * @return the start state, or NULL if it wasn't set.
   */
  state_type* getStartState() {return _start_state;}

  /** Get the set of final (accepting) states in the NBA */
  BitSet& getFinalStates() {return _final_states;}


  BitSet calculateFinalTrueLoops(SCCs& sccs);
  void removeRedundantFinalStates(SCCs& sccs);


  void print(std::ostream& out);
  void print_lbtt(std::ostream& out);
  void print_hoa(std::ostream& out);
  void print_hoa(std::ostream& out, const std::string& tool, const std::string& tool_version);
  void print_dot(std::ostream& out);

  /** Return number of states. */
  std::size_t getStateCount() {return _state_count;}

  /** Set stutter sensitiveness information for this automaton */
  void setStutterSensitivenessInformation(StutterSensitivenessInformation::ptr stutter_information) {
    _stutter_information = stutter_information;
  }

  /** Return the (optional) stutter sensitiveness information for this automaton */
  StutterSensitivenessInformation::ptr getStutterSensitivenessInformation() {return _stutter_information;}

  /** Set the name of the automaton (e.g., for output in a HOA file) */
  void setName(const std::string& name) {
    _name = name;
  }

  /** Get the name of the automaton (e.g., for output in a HOA file) */
  std::string getName() const {
    return _name;
  }

  // -- NBA_I virtual functions -----------

  virtual void nba_i_switchAPSet(APSet_cp apset) {
    _apset = apset;
  }

  /** 
   * Create a new state.
   * @return the index of the new state
   */
  virtual std::size_t nba_i_newState() { 
    return newState()->getName();
  }

  /**
   * Add an edge from state <i>from</i> to state <i>to</i>
   * for the edges covered by the APMonom.
   * @param from the index of the 'from' state
   * @param m the APMonom
   * @param to the index of the 'to' state
   */
  virtual void nba_i_addEdge(std::size_t from,
			     APMonom& m,
			     std::size_t to) {
    (*this)[from]->addEdge(m, *((*this)[to]));
  }

  /**
   * Get the underlying APSet 
   * @return a const pointer to the APSet
   */
  virtual APSet_cp nba_i_getAPSet() {
    return getAPSet_cp();
  }

  /** 
   * Set the final flag (accepting) for a state.
   * @param state the state index
   * @param final the flag
   */
  virtual void nba_i_setFinal(std::size_t state, bool final) {
    (*this)[state]->setFinal(final);
  }

  /**
   * Set the state as the start state.
   * @param state the state index
   */
  virtual void nba_i_setStartState(std::size_t state) {
    setStartState((*this)[state]);
  }

  /** Set the name for the automaton */
  virtual void nba_i_setName(const std::string& name) {
    setName(name);
  }

  /** Type of a boost::shared_ptr for this NBA */
  typedef boost::shared_ptr< NBA<Label,EdgeContainer> > shared_ptr;


  bool isDeterministic();

  void remove(const BitSet& states_to_remove);

  static 
  boost::shared_ptr<NBA<Label, EdgeContainer> >
  product_automaton(NBA<Label, EdgeContainer>& nba_1, NBA<Label, EdgeContainer>& nba_2);

private:
  /** Number of states */
  int _state_count;
  
  /** Storage for the states */
  Index<state_type> _index;

  /** The underlying APSet */
  APSet_cp _apset;
  
  /** The start states */
  state_type *_start_state;

  /** The states that are accepting (final) */
  BitSet _final_states;

  /** Information about the stutter sensitiveness */
  StutterSensitivenessInformation::ptr _stutter_information;

  /** A name for the automaton (e.g., provided by a HOA file) */
  std::string _name;
};


/**
 * Constructor.
 * @param apset The underlying APSet
 */
template <typename Label, template <typename N> class EdgeContainer >
NBA<Label, EdgeContainer>::NBA(APSet_cp apset) 
  : _state_count(0), _apset(apset), _start_state(0) {
}

/**
 * Destructor.
 */
template <typename Label, template <typename N> class EdgeContainer >
NBA<Label, EdgeContainer>::~NBA() {
  for (std::size_t i=0;i<_index.size();i++) {
    if (_index[i]) {
      delete _index[i];
    }
  }
}


/**
 * Add a new state.
 * @return a pointer to the newly generated state
 */
template <typename Label, template <typename N> class EdgeContainer >
typename NBA<Label, EdgeContainer>::state_type* 
NBA<Label, EdgeContainer>::newState() {
  _state_count++;
  state_type *state=new state_type(*this);
  
  _index.add(state);
  return state;
}

/**
 * Print the NBA on the output stream.
 */
template <typename Label, template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::print(std::ostream& out) {
  for (iterator state_it=begin();
       state_it!=end();
       ++state_it) {
    state_type& state=*state_it;
    out << "State " << state.getName();
    if (getStartState()==&state) {
      out << " *";
    }
    if (state.isFinal()) {
      out << " !";
    }

    out << std::endl;

    for (edge_iterator edge_it=state.edges_begin();
	 edge_it!=state.edges_end();
	 ++edge_it) {
      edge_type edge=*edge_it;

      APElement label=edge.first;
      BitSet* to_states=edge.second;

      out << " " << label.toString(getAPSet()) << " -> " << *to_states << std::endl;
    }
  }
}

/**
 * Print the NBA on the output stream in LBTT format.
 */
template <typename Label, template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::print_lbtt(std::ostream& out) {
  out << size() << " " << "1" << std::endl; // states sp acc
  
  for (iterator state_it=begin();
       state_it!=end();
       ++state_it) {
    state_type& state=*state_it;
    out << state.getName();
    if (getStartState()==&state) {
      out << " 1";
    } else {
      out << " 0";
    }
    if (state.isFinal()) {
      out << " 0";
    }

    out << " -1"<< std::endl;

    for (edge_iterator edge_it=state.edges_begin();
	 edge_it!=state.edges_end();
	 ++edge_it) {
      edge_type edge=*edge_it;

      APElement label=edge.first;
      BitSet* to_states=edge.second;

      for (BitSetIterator to_it(*to_states);
           to_it!=BitSetIterator::end(*to_states);
           ++to_it) {
	out << *to_it << " " << label.toStringLBTT(getAPSet()) << std::endl;
      }
    }
    out << "-1" << std::endl;
  }
}


/**
 * Print the NBA on the output stream in HOA format.
 */
template <typename Label, template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::print_hoa(std::ostream& out) {
  print_hoa(out, "");
}

/**
 * Print the NBA on the output stream in HOA format.
 */
template <typename Label, template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::print_hoa(std::ostream& out,
				     const std::string& tool,
				     const std::string& tool_version) {
  out << "HOA: v1" << std::endl;
  out << "States: " << size() << std::endl;
  out << "acc-name: Buchi" << std::endl;
  out << "Acceptance: 1 Inf(0)" << std::endl;
  out << "Start: " << getStartState()->getName() << std::endl;
  if (tool != "") {
    out << "tool: ";
    cpphoafparser::HOAParserHelper::print_quoted(out, tool);
    if (tool_version != "") {
      out << " ";
      cpphoafparser::HOAParserHelper::print_quoted(out, tool_version);
    }
    out << std::endl;
  }

  if (getName()!="") {
    out << "name: ";
    cpphoafparser::HOAParserHelper::print_quoted(out, getName());
    out << std::endl;
  }

  // Enumerate APSet
  out << "AP: " << getAPSize();
  for (unsigned int ap_i=0;ap_i<getAPSize();++ap_i) {
    out << " ";
    cpphoafparser::HOAParserHelper::print_quoted(out, getAPSet().getAP(ap_i));
  }
  out << std::endl;

  out << "properties: explicit-labels trans-labels no-univ-branch";
  if (getStutterSensitivenessInformation() &&
      getStutterSensitivenessInformation()->isCompletelyInsensitive()) {
    out << " stutter-invariant";
  }
  out << std::endl;
  out << "--BODY--" << std::endl;
  
  for (iterator state_it=begin();
       state_it!=end();
       ++state_it) {
    state_type& state=*state_it;
    out << "State: " << state.getName();
    if (state.isFinal()) {
      out << " {0}";
    }
    out << std::endl;

    for (edge_iterator edge_it=state.edges_begin();
         edge_it!=state.edges_end();
         ++edge_it) {
      edge_type edge=*edge_it;

      APElement label=edge.first;
      BitSet* to_states=edge.second;

      bool have_cached_label=false;
      std::string cached_label;
      for (BitSetIterator to_it(*to_states);
           to_it!=BitSetIterator::end(*to_states);
           ++to_it) {
        if (!have_cached_label) {
          cached_label = label.toStringHOA(getAPSet());
          have_cached_label = true;
        }
        out << cached_label << " " << *to_it << std::endl;
      }
    }
  }
  out << "--END--" << std::endl;
}


/** 
 * Remove states from the set of accepting (final) states when this is redundant.
 * @param sccs the SCCs of the NBA
 */
template <typename Label, template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::removeRedundantFinalStates(SCCs& sccs) {
  for (std::size_t scc=0;
       scc<sccs.countSCCs();
       ++scc) {
    if (sccs[scc].cardinality()==1) {
      std::size_t state_id=sccs[scc].nextSetBit(0);
      state_type *state=(*this)[state_id];

      if (state->isFinal()) {
	if (!sccs.stateIsReachable(state_id, state_id)) {
	  // The state is final and has no self-loop
	  //  -> the final flag is redundant
	  state->setFinal(false);
	  //	  std::cerr << "Removing final flag for " << state_id << std::endl;
	}
      }
    }
  }
}

/**
 * Checks if the NBA is deterministic (every edge has at most one target state).
 */
template <typename Label, 
	  template <typename N> class EdgeContainer >
bool
NBA<Label, EdgeContainer>::isDeterministic() {
  for (iterator state_it=begin();
       state_it!=end();
       ++state_it) {
    state_type& state=*state_it;
    for (edge_iterator edge_it=state.edges_begin();
	 edge_it!=state.edges_end();
	 ++edge_it) {
      edge_type edge=*edge_it;
      if (edge.second->cardinality()>1) {
	return false;
      }
    }
  }
  return true;
}

/**
 * Print the NBA in DOT format to the output stream.
 * @param out the output stream 
 */
template <typename Label, 
	  template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::print_dot(std::ostream& out) {

  if (this->getStartState()==0) {
    // No start state! 
    THROW_EXCEPTION(IllegalArgumentException, "No start state in NBA!");
  }


  #define DOT_STATE_FONT "Helvetica"
  #define DOT_EDGE_FONT  "Helvetica"

  out << "digraph nba {\n";
#ifdef DOT_STATE_FONT
  out << " node [fontname=" << DOT_STATE_FONT << "]\n";
#endif

#ifdef DOT_EDGE_FONT
  out << " edge [constraints=false, fontname=" << DOT_EDGE_FONT << "]\n";
#endif

  const APSet& ap_set=getAPSet();

  for (iterator state_it=begin();
       state_it!=end();
       ++state_it) {
    state_type& state=*state_it;
    
    std::size_t i_state=state.getName();
    
    out << "\"" << i_state << "\" [";
    out << "label= \"" << i_state << "\"";

    if (state.isFinal()) {
      out << ", shape=box";
    } else {
      out << ", shape=circle";
    }
    
    if (getStartState()==&state) {
      out << ", style=filled, color=black, fillcolor=grey";
    }
    

    out << "]\n"; // close parameters for state


    // transitions

    for (edge_iterator edge_it=state.edges_begin();
	 edge_it!=state.edges_end();
	 ++edge_it) {
      edge_type edge=*edge_it;
      
      APElement label=edge.first;
      BitSet* to_states=edge.second;

      for (BitSetIterator to_it(*to_states);
           to_it!=BitSetIterator::end(*to_states);
           ++to_it) {

	out << "\"" << i_state << "\" -> \"" << *to_it;
	out << "\" [label=\" " << label.toString(ap_set, false) << "\"]\n";
      }
    }
  }

  out << "}" << std::endl;
}



template <typename Label, 
	  template <typename N> class EdgeContainer >
boost::shared_ptr<NBA<Label, EdgeContainer> >
NBA<Label, EdgeContainer>::product_automaton(NBA<Label, EdgeContainer>& nba_1,
					     NBA<Label, EdgeContainer>& nba_2) {
  assert(nba_1.getAPSet() == nba_2.getAPSet());
  boost::shared_ptr<NBA<Label, EdgeContainer> > product_nba(new NBA<Label, EdgeContainer>(nba_1.getAPSet_cp()));
  
  const APSet& apset=nba_1.getAPSet();
  assert(apset == nba_2.getAPSet());

  for (std::size_t s_1=0;s_1<nba_1.size();s_1++) {
    for (std::size_t s_2=0;s_2<nba_2.size();s_2++) {
      for (std::size_t copy=0; copy<2; copy++) {
	std::size_t s_r=product_nba->nba_i_newState();
	assert(s_r == (s_1*nba_2.size() + s_2)*2 + copy);
#ifdef VERBOSE
	std::cerr << s_1 << ":" << s_2 << ":" << copy << " = " << s_r << std::endl;
#endif

	std::size_t to_copy=copy;
	
	if (copy==0 && nba_1[s_1]->isFinal()) {
	  to_copy=1;
	}
	if (copy==1 && nba_2[s_2]->isFinal()) {
	  (*product_nba)[s_r]->setFinal(true);
	  to_copy=0;
	}
	
	for (typename APSet::element_iterator it=apset.all_elements_begin();
	     it!=apset.all_elements_end();
	     ++it) {
	  APElement label=*it;
	  BitSet *to_s1=nba_1[s_1]->getEdge(label);
	  BitSet *to_s2=nba_2[s_2]->getEdge(label);
	  BitSet to_set;
	  for (BitSetIterator it_e_1=BitSetIterator(*to_s1);
	       it_e_1!=BitSetIterator::end(*to_s1);
	       ++it_e_1) {
	    for (BitSetIterator it_e_2=BitSetIterator(*to_s2);
		 it_e_2!=BitSetIterator::end(*to_s2);
		 ++it_e_2) {
	      std::size_t to=((*it_e_1)*nba_2.size() + (*it_e_2))*2 + to_copy;
	      to_set.set(to);
	    }
	  }
	  
	  *((*product_nba)[s_r]->getEdge(label))=to_set;
	}
      }
    }
  }
  
  std::size_t start_1 = nba_1.getStartState()->getName();
  std::size_t start_2 = nba_2.getStartState()->getName();
  product_nba->setStartState( (*product_nba)[ start_1*nba_2.size() + start_2 ]);
  
  return product_nba;
}

/**
 * Remove a set of states from the NBA. The start state can not
 * be removed.
 */
template <typename Label,
	  template <typename N> class EdgeContainer >
void
NBA<Label, EdgeContainer>::remove(const BitSet& states_to_remove) {

  BitSet to_remove(states_to_remove);
  if (_start_state && states_to_remove.get(_start_state->getName())) {
    // don't clear start state
    to_remove.clear(_start_state->getName());
  }

  // remove states in the index
  for (std::size_t state : to_remove) {
    _index.remove(state);
  }

  // remove states from To bitsets of all edges
  for (std::size_t state=0, n=size(); state < n; state++) {
    if (!(*this)[state])
      // ignore removed states
      continue;

    for (edge_type edge : (*this)[state]->edges()) {
      // label = edge.first
      // to = edge.second

      if (edge.second) {
	// clear all the bits corresponding to removed states
	edge.second->Minus(to_remove);
      }
    }
  }

  // remove states from set of final states
  _final_states.Minus(to_remove);

  // compact the index
  std::pair<bool, std::vector<std::size_t> > mapping = _index.compact();

  if (mapping.first) {
    // some states have moved, we have to map the target sets of all edges

    for (std::size_t state=0, n=size(); state < n; state++) {
      for (edge_type edge : (*this)[state]->edges()) {
	// label = edge.first
	// to = edge.second

	if (edge.second) {
	  edge.second->map(mapping.second);
	}
      }
    }

    // and we map the final states as well...
    _final_states.map(mapping.second);
  }

  // set new state count
  _state_count = _index.size();
}

#endif
