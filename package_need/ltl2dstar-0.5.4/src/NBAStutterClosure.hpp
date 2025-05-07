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


#ifndef NBASTUTTERCLOSURE_HPP
#define NBASTUTTERCLOSURE_HPP

/** @file 
 * Provides NBAStutterClosure.
 */

#include "GraphAlgorithms.hpp"
#include "NBAAnalysis.hpp"

#include "APElement.hpp"

#include <boost/shared_ptr.hpp>


/**
 * Calculate the stutter closure for an NBA.
 */
class NBAStutterClosure {
public:

  /** Calculate the stutter closure for the NBA, for a certain symbol.
   * @param nba the NBA
   * @param label the symbol for which to perform the stutter closure
   */
  template<typename NBA_t>
  static boost::shared_ptr<NBA_t> stutter_closure(NBA_t& nba, APElement label) {
    APSet_cp apset=nba.getAPSet_cp();
    
    boost::shared_ptr<NBA_t> nba_result_ptr(new NBA_t(apset));
    NBA_t& result=*nba_result_ptr;
    
    unsigned int element_count=apset->powersetSize();
    std::size_t V=nba.size();

    std::vector<BitSet> reachable;
    reachable_states(nba, label, reachable);

    assert(nba.getStartState());
    std::size_t start_state=nba.getStartState()->getName();
    
    for (std::size_t i=0;i<V;i++) {
      result.nba_i_newState();
      result.nba_i_newState();
      result.nba_i_newState();
    }

    result.setStartState(result[start_state]);

    for (std::size_t i=0;i<V;i++) {
      if (nba[i]->isFinal()) {
	result[i]->setFinal(true);
      }
      result[i+V]->setFinal(true);

      result[i+2*V]->getEdge(label)->set(i+2*V, true);
      result[i+2*V]->getEdge(label)->set(i, true);

      typename NBA_t::state_type* from=result[i];
      typename NBA_t::state_type* from_F=result[i+V];
      
      for (unsigned int j=0;j<element_count;j++) {
	BitSet& result_to=*(from->getEdge(j));
	
	BitSet* to=nba[i]->getEdge(j);
	if (j!=label) {
	  result_to=*to;
	} else {
	  for (BitSetIterator it=BitSetIterator(*to);
	       it!=BitSetIterator::end(*to);
	       ++it) {
	    std::size_t to_state=*it;
	    
	    // We can go directly to the original state
	    result_to.set(to_state);
	    // We can also go to the corresponding stutter state instead
	    result_to.set(to_state+2*V);
	    
	    // ... and then we can go directly to all the states
	    // that are j-reachable from to
	    result_to.Union(reachable[to_state]);
	  }
	}
	
	// copy result edge states to the _F copy of the state
	*(from_F->getEdge(j)) = result_to;
      }
    }

    return nba_result_ptr;
  }



private:

  template <typename NBA_t>
  static
  void reachable_states(NBA_t& nba, 
			APElement label, 
			std::vector<BitSet>& result) {
    std::size_t V=nba.size();

    APSet_cp empty_apset(new APSet());
    NBA_t r(empty_apset);

    assert(nba.getStartState());
    std::size_t start_state=nba.getStartState()->getName();
    
    for (std::size_t i=0;i<V;i++) {
      r.nba_i_newState();
      r.nba_i_newState();
    }

    r.setStartState(r[start_state]);

    for (std::size_t i=0;i<V;i++) {
      bool i_final=nba[i]->isFinal();
      r[i]->setFinal(i_final);
      r[i+V]->setFinal(true);
      
      BitSet& to_set=*(nba[i]->getEdge(label));
      
      BitSet& r_to_set=*(r[i]->getEdge(0));
      BitSet& rF_to_set=*(r[i+V]->getEdge(0));
      r_to_set=to_set;
      
      for (BitSetIterator bsi=BitSetIterator(to_set);
	   bsi!=BitSetIterator::end(to_set);
	   ++bsi) {
	rF_to_set.set(*bsi+V);
	if (i_final) {
	  r_to_set.set(*bsi+V);
	}
      }
    }

    SCCs scc;
    GraphAlgorithms<NBA_t>::calculateSCCs(r,
					  scc,
					  true);
    result=*scc.getReachabilityForAllStates();
  }
 
};


#endif
