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


#ifndef RABINACCEPTANCE_H
#define RABINACCEPTANCE_H

/** @file
 * Provides class RabinAcceptance.
 */

#include "common/BitSet.hpp"
#include "common/BitSetIterator.hpp"
#include "common/Exceptions.hpp"
#include "common/helper.hpp"
#include <boost/iterator/filter_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <iostream>
#include <string>
#include <vector>


/**
 * Class storing a Rabin acceptance condition.
 * Contains a number k of pairs of BitSets (L_i,U_i), where the states
 * in the acceptance sets L_i and U_i are set.
 *
 * Semantics: A pair (L_i,U_i) is accepting iff L_i is visited infinitely
 * often and U_i is visited at most finitely often:
 *     (G F L_i) & (F G !U_i)
 *
 * The acceptance condition is accepting if at least one of the pairs
 * is accepting.
 */
class RabinAcceptance {
public:
  /**
   * Constructor
   * @param number_of_initial_pairs The initial numbers of pairs to allocate
   */
  RabinAcceptance(std::size_t number_of_initial_pairs=0) :
    _is_compact(true) {
    if (number_of_initial_pairs>0) {
      newAcceptancePairs(number_of_initial_pairs);
    }
  }
  
  /** Destructor */
  ~RabinAcceptance() {
    // delete BitSets in the acceptance pairs
    for (std::vector<BitSet*>::iterator it=_acceptance_L.begin();
         it!=_acceptance_L.end();
         ++it) {
      delete *it;
    }
    
    for (std::vector<BitSet*>::iterator it=_acceptance_U.begin();
         it!=_acceptance_U.end();
         ++it) {
      delete *it;
    }
  }

private:
  RabinAcceptance(const RabinAcceptance& other); // not yet implemented

public:

  
  // ---- The AcceptanceCondition Interface
  
  /**
   * Check if this RabinAcceptance is compact (part of interface AcceptanceCondition).
   * @return true iff compact
   */
  bool isCompact() const {
    return _is_compact;
  }


  /**
   * Make this RabinAcceptance compact (part of interface AcceptanceCondition).
   */
  void makeCompact() {
    if (isCompact()) {
      return;
    }

    // Compress Acceptance-Pairs 
    std::size_t pair_to=0;
    for (std::size_t pair_from=0;
         pair_from<_acceptance_L.size();
         pair_from++) {
      if (_acceptance_L[pair_from]!=0) {
        if (pair_from==pair_to) {
          // nothing to do
        } else {
          _acceptance_L[pair_to]=_acceptance_L[pair_from];
          _acceptance_U[pair_to]=_acceptance_U[pair_from];
        }
        
        ++pair_to;
      }
    }
   
    std::size_t new_acceptance_count=pair_to;
    
    _acceptance_L.resize(new_acceptance_count);
    _acceptance_U.resize(new_acceptance_count);

    _is_compact=true;
  }


  /** Update the acceptance condition upon renaming of states acording
   *  to the mapping (part of AcceptanceCondition interface).
   *  Assumes that states can only get a lower name.
   * @param mapping vector with mapping a[i] -> j
   */
  void moveStates(std::vector<std::size_t> mapping) {
    if (!isCompact()) {
      makeCompact();
    }

    for (std::size_t i=0;i<size();++i) {
      move_acceptance_bits(*_acceptance_L[i], mapping);
      move_acceptance_bits(*_acceptance_U[i], mapping);
    }
  }

  /**
   * Print the Acceptance-Pairs: header (part of interface AcceptanceCondition).
   * @param out the output stream.
   */
  void outputAcceptanceHeader(std::ostream& out) const {
    out << "Acceptance-Pairs: "<< size() << std::endl;
  }

  /**
   * Print the Acc-Sig: line for a state (part of interface AcceptanceCondition).
   * @param out the output stream.
   * @param state_index the state
   */
  void outputAcceptanceForState(std::ostream& out, std::size_t state_index) const {
    out << "Acc-Sig:";
    for (std::size_t pair_index=0;pair_index<size();pair_index++) {
      if (isStateInAcceptance_L(pair_index, state_index)) {
        out << " +" << pair_index;
      }

      if (isStateInAcceptance_U(pair_index, state_index)) {
        out << " -" << pair_index;
      }
    }
    out << std::endl;
  }


  /**
   * Add a state (part of interface AcceptanceCondition).
   * @param state_index the index of the added state.
   */
  void addState(std::size_t state_index) {
    UNUSED(state_index);
    // TODO: Assert that state_index > highest set bit
    ;
  }

  /** The 3 different colors for RabinAcceptance */
  enum RabinColor {RABIN_WHITE=0, RABIN_GREEN=1, RABIN_RED=2};

  /** A class storing the acceptance signature for a state
   * (for every acceptance pair one color).
   */
  class RabinSignature {
  public:
    /** Constructor 
     * @param size the number of acceptance pairs 
     */
    RabinSignature(std::size_t size) : _size(size) {}

    /** Constructor
     * @param other another RabinSignature
     */
    RabinSignature(const RabinSignature& other) : 
      _L_bits(other._L_bits), _U_bits(other._U_bits), _size(other._size) {}

    /** Constructor
     * @param L the L part of the acceptance signature.
     * @param U the U part of the acceptance signature.
     * @param size the number of acceptance pairs
     */
    RabinSignature(const BitSet& L_bits, const BitSet U_bits, std::size_t size) 
      : _L_bits(L_bits), _U_bits(U_bits), _size(size) {}
    
    /** Constructor for getting the acceptance signature for a Tree.
     * @param tree the Tree, get acceptance signature from 
     *    tree->generateAcceptance(*this).
     */
    template <typename Tree>
    RabinSignature(const boost::shared_ptr<Tree>& tree) : _size(0) {
      tree->generateAcceptance(*this);
    }

    /** Clear the acceptance signature */
    void clear() {
      _L_bits.clear();
      _U_bits.clear();
    }

    /** Get the L part of this acceptance signature */
    const BitSet& getL() const {return _L_bits;}    
    /** Get the U part of this acceptance signature */
    const BitSet& getU() const {return _U_bits;}

    /** Get the L part of this acceptance signature */
    BitSet& getL() {return _L_bits;}
    /** Get the U part of this acceptance signature */
    BitSet& getU() {return _U_bits;}

    /** Set index to value in the L part of this acceptance signature. */
    void setL(std::size_t index, bool value=true) {
      _L_bits.set(index, value);
    }

    /** Set index to value. in the U part of this acceptance signature. */
    void setU(std::size_t index, bool value=true) {
      _U_bits.set(index, value);
    }

    /** Set the L and U parts according to RabinColor c. 
     * @param i The pair index
     * @param c the RabinColor
     */
    void setColor(std::size_t i, RabinColor c) {
      switch (c) {
      case RABIN_RED:
	_U_bits.set(i, true);
	_L_bits.set(i, false);
	break;

      case RABIN_GREEN:
	_U_bits.set(i, false);
	_L_bits.set(i, true);
	break;

      case RABIN_WHITE:
	_U_bits.set(i, false);
	_L_bits.set(i, false);
	break;
      }
    };

    /** Get the RabinColor for a pair i */
    RabinColor getColor(std::size_t i) const {
      return _U_bits.get(i) ? RABIN_RED : (_L_bits.get(i) ? RABIN_GREEN : RABIN_WHITE);
    }

    /** Get string representation of this signature. */
    std::string toString() const {
      std::string a;
      a="{";
      for (std::size_t i=0;i<size();i++) {
	switch (getColor(i)) {
	case RABIN_RED:
	  a+='-'+boost::lexical_cast<std::string>(i);
	  break;
	case RABIN_GREEN:
	  a+='+'+boost::lexical_cast<std::string>(i);
	  break;
	case RABIN_WHITE:
	  break;
	}
      }
      a+="}";
      
      return a;
    }

    /** Compare to other signature for equality. */
    bool operator==(const RabinSignature& other) const {
      return ( _L_bits==other.getL() ) && ( _U_bits==other.getU() );
    }

    /** Compare less_than to other signature */
    bool operator<(const RabinSignature& other) const {
      if (_L_bits<other.getL()) {
	return true;
      } else if (_L_bits==other.getL()) {
	return _U_bits<other.getU();
      }
      return false;
    }

    /** Get the number of acceptance pairs */
    std::size_t getSize() const {return _size;}
    /** Get the number of acceptance pairs */
    std::size_t size() const {return _size;}

    /** Set the number of acceptance pairs */
    void setSize(std::size_t size) {_size=size;}

    /** Merge this acceptance signature with other signature,
     *  for each tuple element calculate the maximum of the
     *  colors according to the order 
     * RABIN_WHITE < RABIN_GREEN < RABIN_RED */
    void maxMerge(const RabinSignature& other) {
      for (std::size_t i=0;i<_size;i++) {
	if (getColor(i) < other.getColor(i)) {
	  setColor(i, other.getColor(i));
	}
      }
    }

    /**
     * Calculate a hash value using HashFunction
     * @param hashfunction the HashFunction
     */
    template <class HashFunction>
    void hashCode(HashFunction& hashfunction) {
      _L_bits.hashCode(hashfunction);
      _U_bits.hashCode(hashfunction);
    }
    
    
  private:
    /** The L part */
    BitSet _L_bits;
    /** The U part */
    BitSet _U_bits;    
    /** The number of acceptance pairs */
    std::size_t _size;
  };

  /** The signature_type (part of AcceptanceCondition interface) */
  typedef RabinSignature signature_type;

  /** Accessor for the acceptance signature for a state 
   *  (part of AcceptanceCondition interface)
   */
  class AcceptanceForState {
  private:
    /** Reference to the underlying RabinAcceptance */
    RabinAcceptance& _acceptance;
    /** The state index for this accessor */
    std::size_t _state_index;

  public:
    /** Constructor */
    AcceptanceForState(RabinAcceptance& acceptance,
		       std::size_t state_index) :
      _acceptance(acceptance), _state_index(state_index) {}

    /** Add this state to L[pair_index] */
    void addTo_L(std::size_t pair_index) {
      _acceptance.getAcceptance_L(pair_index).set(_state_index);
      _acceptance.getAcceptance_U(pair_index).set(_state_index, false);
    }
    
    /** Add this state to U[pair_index] */
    void addTo_U(std::size_t pair_index) {
      _acceptance.getAcceptance_U(pair_index).set(_state_index);
      _acceptance.getAcceptance_L(pair_index).set(_state_index, false);
    }

    /** Is this state in L[pair_index] */
    bool isIn_L(std::size_t pair_index) const {
      return _acceptance.isStateInAcceptance_L(pair_index, _state_index);
    }

    /** Is this state in U[pair_index] */
    bool isIn_U(std::size_t pair_index) const {
      return _acceptance.isStateInAcceptance_U(pair_index, _state_index);
    }
    
    /** Set L and U for this state according to RabinSignature */
    void setSignature(const RabinSignature& signature) {
      for (std::size_t i=0;i<signature.size();i++) {
	if (signature.getL().get(i)) {
	  addTo_L(i);
	}
	if (signature.getU().get(i)) {
	  addTo_U(i);
	}
      }
    }

    /** Get number of acceptance pairs */
    std::size_t size() const {return _acceptance.size();}

    /** Get the signature for this state */
    RabinSignature getSignature() const {
      return RabinSignature(_acceptance.getAcceptance_L_forState(_state_index),
			    _acceptance.getAcceptance_U_forState(_state_index),
			    _acceptance.size());
    }
  };



  

  // ---- Rabin/Streett acceptance specific

  /**
   * Creates a new acceptance pair.
   * @return the index of the new acceptance pair.
   */
  std::size_t newAcceptancePair() {
    BitSet *l=new BitSet();
    BitSet *u=new BitSet();
    
    _acceptance_L.push_back(l);
    _acceptance_U.push_back(u);
    
    _acceptance_count++;  
    return _acceptance_L.size()-1;
  }


  /**
   * Creates count new acceptance pairs.
   * @return the index of the first new acceptance pair.
   */
  std::size_t newAcceptancePairs(std::size_t count) {
    std::size_t rv=_acceptance_L.size();
    
    for (std::size_t i=0;i<count;i++) {
      newAcceptancePair();
    }
    
    return rv;
  }
  

  /**
   * Delete an acceptance pair.
   */
  void removeAcceptancePair(std::size_t pair_index) {
    if (_acceptance_L[pair_index]!=0) {
      _acceptance_count--;
    }
    
    delete _acceptance_L[pair_index];
    delete _acceptance_U[pair_index];
    
    _acceptance_L[pair_index]=0;
    _acceptance_U[pair_index]=0;
    
    _is_compact=false;
  }


  /**
   * Get a reference to the BitSet representing L[pair_index], 
   * allowing changes to this set.
   */
  BitSet& getAcceptance_L(std::size_t pair_index) {
    return *_acceptance_L[pair_index];
  }

  /**
   * Get a reference to the BitSet representing U[pair_index], 
   * allowing changes to this set.
   */
  BitSet& getAcceptance_U(std::size_t pair_index) {
    return *_acceptance_U[pair_index];
  }
  

  /**
   * Get the L part of the acceptance signature for a state (changes to the
   * BitSet do not affect the automaton).
   */
  BitSet getAcceptance_L_forState(std::size_t state_index) const {
    BitSet result;
    getBitSetForState(state_index, _acceptance_L, &result);

    return result;
  }

  /**
   * Get the U part of the acceptance signature for a state (changes to the
   * BitSet do not affect the automaton).
   */
  BitSet getAcceptance_U_forState(std::size_t state_index) const {
    BitSet result;
    getBitSetForState(state_index, _acceptance_U, &result);

    return result;
  }

  /** Is a certain state in L[pair_index]? */
  bool isStateInAcceptance_L(std::size_t pair_index,
			     std::size_t state_index) const {
    return _acceptance_L[pair_index]->get(state_index);
  }

  /** Is a certain state in U[pair_index]? */
  bool isStateInAcceptance_U(std::size_t pair_index,
			     std::size_t state_index) const {
    return _acceptance_U[pair_index]->get(state_index);
  }

  /** Set L[pair_index] for this state to value. */
  void stateIn_L(std::size_t pair_index,
		 std::size_t state_index,
		 bool value=true) {
    getAcceptance_L(pair_index).set(state_index,value);
  }

  /** Set U[pair_index] for this state to value. */
  void stateIn_U(std::size_t pair_index,
		 std::size_t state_index,
		 bool value=true) {
    getAcceptance_U(pair_index).set(state_index,value);
  }
  
  /** Get the number of acceptance pairs. 
   *  Requires the acceptance pairs to be compact. */
  std::size_t size() const {
    if (!isCompact()) {
      throw Exception("Can't give acceptance pair count for uncompacted condition.");
    }
    return _acceptance_L.size();
  }


private:
  /** A vector of BitSet* */
  typedef std::vector<BitSet*> BitSetVector;

  /** Helper functor for acceptance_pair_iterator. */
  struct acceptance_is_not_null{ 
    BitSetVector& _acceptance_vector;
    acceptance_is_not_null(BitSetVector& acceptance_vector) :
      _acceptance_vector(acceptance_vector) {;};

    bool operator()(int i) { return _acceptance_vector[i]!=0; };
  };

public:
  /** Type of an iterator over the index of acceptance pairs. */
  typedef boost::filter_iterator<acceptance_is_not_null, 
                                 boost::counting_iterator<std::size_t> >
    acceptance_pair_iterator;

  /** Iterator pointing to the index of the first acceptance pair. */
  acceptance_pair_iterator acceptance_pair_begin() {
    return acceptance_pair_iterator(acceptance_is_not_null(_acceptance_L), boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(_acceptance_L.size()));
  };

  /** Iterator pointing after the index of the last acceptance pair. */
  acceptance_pair_iterator acceptance_pair_end() {
    return acceptance_pair_iterator(acceptance_is_not_null(_acceptance_L), boost::counting_iterator<std::size_t>(_acceptance_L.size()), boost::counting_iterator<std::size_t>(_acceptance_L.size()));
  };


private: //functions 
  /** Calculate the BitSet for a state from the acceptance pairs, store
   *  result in result.
   *  @param state_index the state
   *  @param acc the BitSetVector (either _L or _U)
   *  @param result the Bitset where the results are stored, has to be clear 
   *                at the beginning!
   */
  void getBitSetForState(std::size_t state_index,
			 const BitSetVector& acc,
			 BitSet* result) const {
    
    for (std::size_t i=0;i<acc.size();i++) {
      if (acc[i]!=0) {
	if (acc[i]->get(state_index)) {
	  result->set(i);
	}
      }
    }
  }


  /** 
   * Move the bits set in acc to the places specified by mapping.
   */
  void move_acceptance_bits(BitSet& acc,
			    std::vector<std::size_t> mapping) {
    for (BitSetIterator it(acc);
         it!=BitSetIterator::end(acc);
         ++it) {
      std::size_t i=*it;
      std::size_t j=mapping[i];
      // :: j is always <= i
      if (j>i) {
	THROW_EXCEPTION(Exception, "Wrong mapping in move_acceptance_bits");
      }

      if (i == j) {
	// do nothing
      } else {
	// move bit from i->j
	acc.set(j);
	acc.clear(i);
      }
    }
  }




private: // members
  /** The number of acceptance pairs */
  std::size_t _acceptance_count;

			    
  /** A vector of BitSet* representing the L part of the acceptance pairs. */
  BitSetVector _acceptance_L;
  /** A vector of BitSet* representing the U part of the acceptance pairs. */
  BitSetVector _acceptance_U;

  bool _is_compact;
};







#endif

