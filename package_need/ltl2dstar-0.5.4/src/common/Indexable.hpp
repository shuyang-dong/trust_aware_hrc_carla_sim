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


#ifndef INDEXABLE_H
#define INDEXABLE_H

/** @file
 * Base class for objects that are placed into an Index.
 */

#include "common/Exceptions.hpp"

// include "common/Index.h"

#include <algorithm>

//class Index;

template <typename T> class Index;

/**
 * Base class that implements the interface needed for
 * objects that are placed into an Index.
 * <p>
 * Template parameter T is the object that inherits
 * from this base class: <br>
 * class myObject : public Indexable<myObject> {...
 * </p>
 * <p>
 * An object can be placed in multiple Index containers.
 * </p>
 */
template <typename T>
class Indexable {
 public:
  Indexable();
  ~Indexable();

  std::size_t idx_getIndex(const Index<T> *idx) const;
  void idx_setIndex(Index<T> *idx, std::size_t index);
  void idx_clearIndex(Index<T> *idx);
  bool idx_hasIndex(const Index<T> *idx) const;

 private:
  std::size_t index_count;
  Index<T>** indexes;
  size_t* indizes;

  bool in_destructor;

  typedef or_nil<std::size_t> index_or_nil_t;
  index_or_nil_t find_idx(const Index<T> *idx) const;
};

/**
 * Constructor.
 */
template <typename T>
Indexable<T>::Indexable() {
  indexes=(Index<T>**)0;
  indizes=(std::size_t*)0;
  index_count=0;
  in_destructor=false;
}


/**
 * Destructor.
 */
template <typename T>
Indexable<T>::~Indexable() {
  in_destructor=true;
  if (indexes) {
    for (std::size_t i=0;i<index_count;i++) {
      // Remove this object from index
      indexes[i]->remove(indizes[i]);
    }
    delete[]indexes;
    delete[]indizes;
  }
}

/**
 * Get the ID of the Index corresponding to idx
 * @return the ID of the Index, NIL if the object is not member of *idx
 */
template <typename T>
typename Indexable<T>::index_or_nil_t Indexable<T>::find_idx(const Index<T> *idx) const {
  if (index_count==0) {
    return index_or_nil_t::NIL();
  }
  
  // Binary Search
  std::size_t l=0, r=index_count-1;
  std::size_t m;

  while (l<=r) {
    m=(r+l)/2;
    if (indexes[m] == idx) {
      return m;
    }

    if (idx > indexes[m]) {
      l=m+1;
    } else {
      r=m-1;
    }
  }
  return index_or_nil_t::NIL();
}

/**
 * Get the ID in this object for the Index idx.
 */
template <typename T>
std::size_t Indexable<T>::idx_getIndex(const Index<T> *idx) const {
  index_or_nil_t idx_idx=find_idx(idx);
  if (idx_idx.isNIL()) {
    THROW_EXCEPTION(Exception, "Index not found!");
  }

  return indizes[idx_idx];
}


/**
 * Set the index value for Index idx (on placement of the object into the index)
 */
template <typename T>
void Indexable<T>::idx_setIndex(Index<T> *idx, std::size_t index) {
  index_or_nil_t idx_idx=find_idx(idx);
  if (!idx_idx.isNIL()) {
    indizes[idx_idx]=index;
    return;
  }

  // We have to add this idx
  if (index_count==0) {
    indexes=new Index<T>*[1];
    indizes=new std::size_t[1];
    indexes[0]=idx;
    indizes[0]=index;
    index_count=1;
    return;
  }

  Index<T> **idx_new=new Index<T>*[index_count+1];
  std::size_t* indizes_new=new std::size_t[index_count+1];
  
  bool was_inserted=false;
  std::size_t i=0, j=0;
  while (i<index_count || j<index_count+1) {
    if (!was_inserted && (i==index_count || indexes[i]>idx)) {
      idx_new[j]=idx;
      indizes_new[j]=index;
      was_inserted=true;
      j++;
    } else {
      idx_new[j]=indexes[i];
      indizes_new[j]=indizes[i];
      i++;
      j++;
    }
  }

  delete[] indexes;
  delete[] indizes;
  indexes=idx_new;
  indizes=indizes_new;
  index_count++;
}

/**
 * Removes the stored information in this object for index idx (on removal from idx)
 */
template <typename T>
void Indexable<T>::idx_clearIndex(Index<T> *idx) {
  if (in_destructor) {
    // We are called while destructing this object, so
    // we don't do a thing, because we will be gone anyway
    return;
  }

  index_or_nil_t idx_idx_=find_idx(idx);
  if (idx_idx_.isNIL()) {
    THROW_EXCEPTION(Exception, "Can't find index!");
  }
  std::size_t idx_idx = idx_idx_.getValue();

  if (index_count==1) {
    delete[] indizes;
    delete[] indexes;
    
    indizes=0;
    indexes=0;
    index_count=0;
    return;
  }

  Index<T> **idx_new=new Index<T>*[index_count-1];
  std::size_t* indizes_new=new std::size_t[index_count-1];
  
  std::size_t j=0;
  for (std::size_t i=0;i<index_count;i++) {
    if (i!=idx_idx) {
      idx_new[j]=indexes[i];
      indizes_new[j]=indizes[i];
      j++;
    }
  }

  delete[] indexes;
  delete[] indizes;
  indexes=idx_new;
  indizes=indizes_new;
  index_count--;
}

/**
 * Checks if the object is member of Index idx.
 */
template <typename T>
bool Indexable<T>::idx_hasIndex(const Index<T> *idx) const {
  return !find_idx(idx).isNIL();
}


#endif
