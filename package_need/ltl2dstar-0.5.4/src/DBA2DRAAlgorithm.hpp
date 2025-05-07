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


#ifndef DBA2DRAALGORITHM_HPP
#define DBA2DRAALGORITHM_HPP

/** @file
 * Provides class DBA2DRAAlgorithm, which can convert a
 * deterministic Büchi automaton, i.e., an NBA where every state
 * has at most one successor for each state and label,
 * to a deterministic Rabin automaton (via NBA2DRA or StutteredNBA2DRA).
 */

#include <string>
#include <boost/shared_ptr.hpp>
#include "common/Exceptions.hpp"

/**
 * Provides conversion from deterministic Büchi to deterministic Rabin.
 */
template <typename NBA_t>
class DBA2DRAAlgorithm {
public:

  /**
   * Constructor
   * @param nba the NBA to convert (needs to be deterministic)
   */
  DBA2DRAAlgorithm(NBA_t& nba) : _nba(nba) {
    // sink state index = largest index in NBA + 1
    _sink =_nba.size();
  }

  /** Information about a DBA state, i.e., index and acceptance flag. */
  class dba_state_t {
  public:
    typedef boost::shared_ptr<dba_state_t> ptr;

    /** Constructor, index and acceptance flag */
    dba_state_t(std::size_t index, bool accepting) :
      _index(index), _accepting(accepting) {}

    /** Get the index of this state */
    std::size_t getID() const {
      return _index;
    }

    /** Is this state accepting? */
    bool isAccepting() const {
      return _accepting;
    }

    /** Generate acceptance for this state */
    void generateAcceptance(RabinAcceptance::AcceptanceForState acceptance) const {
      if (isAccepting()) {
	acceptance.addTo_L(0);
      }
    }

    /** Generate acceptance for this state */
    void generateAcceptance(RabinAcceptance::RabinSignature& acceptance) const {
      acceptance.setSize(1);
      if (_accepting) {
        acceptance.setColor(0, RabinAcceptance::RABIN_GREEN);
      } else {
        acceptance.setColor(0, RabinAcceptance::RABIN_WHITE);
      }
    }

    /** Generate acceptance for this state */
    RabinAcceptance::RabinSignature generateAcceptance() const {
      RabinAcceptance::RabinSignature s(1);
      generateAcceptance(s);
      return s;
    }

    /** Generate detailed HTML state information */
    std::string toHTML() {
      return std::to_string(_index) + (_accepting ? " !" : "");
    }

    /**
     * Equality operator. Only takes into account the state index,
     * as acceptance is unique per DBA state
     */
    bool operator==(const dba_state_t& other) const {
      return _index == other._index;
    }

    /**
     * Less-than operator. Only takes into account the state index,
     * as acceptance is unique per DBA state
     */
    bool operator<(const dba_state_t& other) const {
      return _index < other._index;
    }

  /**
   * Calculate a hash value using HashFunction.
   * Only takes into account the state index,
   * as acceptance is unique per DBA state
   * @param hashfunction the HashFunction
   */
  template <class HashFunction>
  void hashCode(HashFunction& hashfunction) {
    hashfunction.hash(_index);
  }

  private:
    /** The DBA state index */
    std::size_t _index;
    /** Is accepting? */
    bool _accepting;
  };

  /** A result wrapper for a dba_state_t */
  class dba_result_t {
  public:
    typedef boost::shared_ptr<dba_result_t> ptr;

    /** Constructor */
    dba_result_t(typename dba_state_t::ptr state) : _state(state) {}

    /** Get the wrapped state */
    typename dba_state_t::ptr getState() {
      return _state;
    }

  private:
    /** The wrapped state */
    typename dba_state_t::ptr _state;
  };

  /** Algorithm state type */
  typedef typename dba_state_t::ptr state_t;
  /** Algorithm result type (here, simply a wrapper around state_t) */
  typedef typename dba_result_t::ptr result_t;

  /** Algorithm method: Is the NBA empty? */
  bool checkEmpty() {
    if (_nba.size()==0 ||
        _nba.getStartState()==0) {
      return true;
    }
    return false;
  }

  /** Algorithm method: Get next state for given dba_state and label elem. */
  result_t delta(state_t dba_state, APElement elem) {
    if (dba_state->getID() == _sink) {
      return result_t(new dba_result_t(dba_state));
    }

    BitSet *to=_nba[dba_state->getID()]->getEdge(elem);

    std::size_t to_cardinality=0;
    if (to!=0) {
          to_cardinality=to->cardinality();
    }

    state_t successor;
    if (to==0 ||
        to_cardinality==0) {
      // empty to -> go to sink state
      successor.reset(new dba_state_t(_sink, false));
    } else if (to_cardinality==1) {
      std::size_t to_index=to->nextSetBit(0);
      successor.reset(new dba_state_t(to_index, _nba[to_index]->isFinal()));
    } else {
      THROW_EXCEPTION(IllegalArgumentException, "NBA is no DBA!");
    }

    return result_t(new dba_result_t(successor));
  }

  /** Algorithm method: Get the start state. */
  state_t getStartState() {
    std::size_t start_index =
      _nba.getStartState()!=0 ? _nba.getStartState()->getName() : _sink;
    bool start_final =
      _nba.getStartState()!=0 ? _nba.getStartState()->isFinal() : false;
    state_t start(new dba_state_t(start_index, start_final));
    return start;
  }

  /** Prepare the acceptance condition (one Rabin pair) */
  void prepareAcceptance(RabinAcceptance& acceptance) {
    acceptance.newAcceptancePairs(1);
  }

private:
  /** The deterministic NBA to be converted */
  NBA_t& _nba;

  /**
   * The index of an additional, non-accepting sink state.
   * This state serves to make the automaton complete.
   */
  std::size_t _sink;
};

#endif
