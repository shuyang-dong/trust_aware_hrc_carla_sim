/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_PROMELA_NBA_PARSER_PROMELA_TAB_HPP_INCLUDED
# define YY_PROMELA_NBA_PARSER_PROMELA_TAB_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int promela_debug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    PROMELA_AP = 258,
    PROMELA_OR = 259,
    PROMELA_AND = 260,
    PROMELA_NOT = 261,
    PROMELA_TRUE = 262,
    PROMELA_FALSE = 263,
    PROMELA_NEVER = 264,
    PROMELA_IF = 265,
    PROMELA_FI = 266,
    PROMELA_DO = 267,
    PROMELA_OD = 268,
    PROMELA_GOTO = 269,
    PROMELA_SKIP = 270,
    PROMELA_ASSERT = 271,
    PROMELA_ATOMIC = 272,
    PROMELA_LABEL = 273,
    PROMELA_COLON = 274,
    PROMELA_SEMICOLON = 275,
    PROMELA_DOUBLE_COLON = 276,
    PROMELA_LBRACE = 277,
    PROMELA_RBRACE = 278,
    PROMELA_LPAREN = 279,
    PROMELA_RPAREN = 280,
    PROMELA_RIGHT_ARROW = 281
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE promela_lval;

int promela_parse (void);

#endif /* !YY_PROMELA_NBA_PARSER_PROMELA_TAB_HPP_INCLUDED  */
