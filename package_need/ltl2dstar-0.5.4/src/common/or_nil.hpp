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


#ifndef OR_NIL_H
#define OR_NIL_H

/** @file
 * Wrapper class around a value to allow for the possibility
 * that there is no value (NIL)
 */
template <class Value>
class or_nil {
  Value _value;
  bool _is_nil;

public:

  or_nil() : _is_nil(true) {}
  or_nil(const Value& value) : _value(value), _is_nil(false) {}

  static or_nil NIL() {return or_nil();}

  bool isNIL() {return _is_nil;}
  operator Value() const {
    if (_is_nil) {
      THROW_EXCEPTION(IllegalArgumentException, "Can not convert NIL!");
    }

    return _value;
  }

  Value getValue() const {
    return (Value)*this;
  }

  bool operator==(const or_nil& other) const {
    if (_is_nil || other._is_nil) {
      return _is_nil == other._is_nil;
    } else {
      return _value == other._value;
    }
  }
};


#endif

