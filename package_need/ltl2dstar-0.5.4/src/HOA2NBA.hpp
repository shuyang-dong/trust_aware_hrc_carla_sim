#ifndef HOA2NBA_H
#define HOA2NBA_H

#include "cpphoafparser/parser/hoa_parser.hh"
#include "NBA_I.hpp"
#include <vector>

class HOA2NBA : public cpphoafparser::HOAConsumer {
public:
  typedef std::shared_ptr<HOA2NBA> ptr;

  /**
   * Parse a single HOA automaton (NBA) from in and store in nba.
   * The properties of the automaton are stored in the provided vector hoa_properties.
   * Optionally, a target APSet can be provided, with the APs of the HOA automaton
   * translated to the target APSet.
   */
  static bool parse(std::istream& in,
		    NBA_I* nba,
		    std::vector<std::string>& hoa_properties,
		    APSet_cp target_apset=APSet_cp()) {
    HOA2NBA::ptr consumer(new HOA2NBA(nba, hoa_properties, target_apset));
    
    try {
      cpphoafparser::HOAParser::parse(in, consumer);
      
      return true;
    } catch (cpphoafparser::HOAParserException& e) {
      std::cerr << e.what() << std::endl;
    } catch (cpphoafparser::HOAConsumerException& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
    }
    return false;
  }

  /** 
   * This function is called by the parser to query the consumer whether aliases should be
   * resolved by the parser (return `true` or whether the consumer would like to
   * see the aliases unresolved (return `false`). This function should always return
   * a constant value.
   **/
  virtual bool parserResolvesAliases() override {
    return true;
  }

  /** Called by the parser for the "HOA: version" item [mandatory, once]. */
  virtual void notifyHeaderStart(const std::string& version) override {
  }

  /** Called by the parser for the "States: int(numberOfStates)" item [optional, once]. */
  virtual void setNumberOfStates(unsigned int numberOfStates) override {
    _knowNumberOfStates = true;
    _hoaNumberOfStates = numberOfStates;
  }

  /** 
   * Called by the parser for each "Start: state-conj" item [optional, multiple]. 
   * @param stateConjunction a list of state indizes, interpreted as a conjunction
   **/
  virtual void addStartStates(const int_list& stateConjunction) override {
    if (stateConjunction.size() != 1) {
      throw cpphoafparser::HOAConsumerException("Automaton has universal branching, not supported");
    }

    if (_haveStartState) {
      throw cpphoafparser::HOAConsumerException("Currently, only a single initial state is supported");
    }

    _startState = stateConjunction.front();
    _haveStartState = true;
  }

  /**
   * Called by the parser for each "Alias: alias-def" item [optional, multiple].
   * Will be called no matter the return value of `parserResolvesAliases()`.
   * 
   *  @param name the alias name (without @)
   *  @param labelExpr a boolean expression over labels
   **/
  virtual void addAlias(const std::string& name, label_expr::ptr labelExpr) override {
    // nothing to do, aliases already resolved
  }

  /**
   * Called by the parser for the "AP: ap-def" item [optional, once].
   * @param aps the list of atomic propositions
   */
  virtual void setAPs(const std::vector<std::string>& aps) override {
    for (const std::string& ap : aps) {
      _apset->addAP(ap);
    }
  }

  /**
   * Called by the parser for the "Acceptance: acceptance-def" item [mandatory, once].
   * @param numberOfSets the number of acceptance sets used to tag state / transition acceptance
   * @param accExpr a boolean expression over acceptance atoms
   **/
  virtual void setAcceptanceCondition(unsigned int numberOfSets, acceptance_expr::ptr accExpr) {
    if (numberOfSets == 1
	&& accExpr->getType() == acceptance_expr::EXP_ATOM
	&& accExpr->getAtom().getType() == cpphoafparser::AtomAcceptance::TEMPORAL_INF
	&& accExpr->getAtom().getAcceptanceSet() == 0
	&& !accExpr->getAtom().isNegated()) {
      _isBuchi = true;
    } else if (numberOfSets == 0
	       && accExpr->getType() == acceptance_expr::EXP_TRUE) {
      _isAllAccepting = true;
    } else if (numberOfSets == 0
	       && accExpr->getType() == acceptance_expr::EXP_FALSE) {
      _isNoneAccepting = true;
    } else {
      throw cpphoafparser::HOAConsumerException("Expected Buchi automaton, e.g., 'Acceptance: 1 Inf(0)', found 'Acceptance: "+std::to_string(numberOfSets)+" "+accExpr->toString()+"'");
    }
  }

  /** 
   * Called by the parser for each "acc-name: ..." item [optional, multiple].
   * @param name the provided name
   * @param extraInfo the additional information for this item
   * */
  virtual void provideAcceptanceName(const std::string& name, const std::vector<cpphoafparser::IntOrString>& extraInfo) override {
    // ignore
  }

  /**
   * Called by the parser for the "name: ..." item [optional, once].
   **/
  virtual void setName(const std::string& name) override {
    _nba->nba_i_setName(name);
  }

  /**
   * Called by the parser for the "tool: ..." item [optional, once].
   * @param name the tool name
   * @param version the tool version (option, empty pointer if not provided)
   **/
  virtual void setTool(const std::string& name, std::shared_ptr<std::string> version) override {
  }

  /**
   * Called by the parser for the "properties: ..." item [optional, multiple].
   * @param properties a list of properties
   */
  virtual void addProperties(const std::vector<std::string>& properties) override {
    _properties.insert(_properties.end(), properties.cbegin(), properties.cend());
  }

  /** 
   * Called by the parser for each unknown header item [optional, multiple].
   * @param name the name of the header (without ':')
   * @param content a list of extra information provided by the header
   */
  virtual void addMiscHeader(const std::string& name, const std::vector<cpphoafparser::IntOrString>& content) override {
    if (name.at(0) >= 'A' && name.at(0) <= 'Z') {
      // reject headers that have semantic meaning
      throw cpphoafparser::HOAConsumerException("Unknown header " + name);
    }
  }

  /**
   * Called by the parser to notify that the BODY of the automaton has started [mandatory, once].
   */
  virtual void notifyBodyStart() override {
    if (!(_isBuchi || _isAllAccepting || _isNoneAccepting)) {
      throw cpphoafparser::HOAConsumerException("Require Buchi automaton");
    }

    if (_target_apset) {
      generateAPMapping();
      _nba->nba_i_switchAPSet(_target_apset);
    } else {
      _nba->nba_i_switchAPSet(_apset);
    }

    if (_knowNumberOfStates && _hoaNumberOfStates > 0) {
      createStatesUpTo(_hoaNumberOfStates-1);
    }
  }

  /** 
   * Called by the parser for each "State: ..." item [multiple]. 
   * @param id the identifier for this state
   * @param info an optional string providing additional information about the state (empty pointer if not provided)
   * @param labelExpr an optional boolean expression over labels (state-labeled) (empty pointer if not provided)
   * @param accSignature an optional list of acceptance set indizes (state-labeled acceptance) (empty pointer if not provided)
   */
  virtual void addState(unsigned int id, std::shared_ptr<std::string> info, label_expr::ptr labelExpr, std::shared_ptr<int_list> accSignature) override {
    if (labelExpr) {
      throw cpphoafparser::HOAConsumerException("Only support labels on transitions");
    }

    createStatesUpTo(id);

    if (accSignature) {
      for (unsigned int accSet : *accSignature) {
	if (accSet == 0) {
	  _nba->nba_i_setFinal(id, true);
	}
      }
    }

  }

  /** 
   * Called by the parser for each implicit edge definition [multiple], i.e.,
   * where the edge label is deduced from the index of the edge.
   *
   * If the edges are provided in implicit form, after every `addState()` there should be 2^|AP| calls to
   * `addEdgeImplicit`. The corresponding boolean expression over labels / BitSet
   * can be obtained by calling BooleanExpression.fromImplicit(i-1) for the i-th call of this function per state. 
   * 
   * @param stateId the index of the 'from' state
   * @param conjSuccessors a list of successor state indizes, interpreted as a conjunction 
   * @param accSignature an optional list of acceptance set indizes (transition-labeled acceptance) (empty pointer if not provided)
   */
  virtual void addEdgeImplicit(unsigned int stateId, const int_list& conjSuccessors, std::shared_ptr<int_list> accSignature) override {
    throw cpphoafparser::HOAConsumerException("Currently do not support implicit edges");
  }

  /**
   * Called by the parser for each explicit edge definition [optional, multiple], i.e.,
   * where the label is either specified for the edge or as a state-label.
   * <br/>
   * @param stateId the index of the 'from' state
   * @param labelExpr a boolean expression over labels (empty pointer if none provided, only in case of state-labeled states)
   * @param conjSuccessors a list of successors state indizes, interpreted as a conjunction 
   * @param accSignature an optional list of acceptance set indizes (transition-labeled acceptance) (empty pointer if none provided)
   */
  virtual void addEdgeWithLabel(unsigned int stateId, label_expr::ptr labelExpr, const int_list& conjSuccessors, std::shared_ptr<int_list> accSignature) override {
    if (accSignature) {
      throw cpphoafparser::HOAConsumerException("Do not support transition-based acceptance");
    }

    if (conjSuccessors.size() != 1) {
      throw cpphoafparser::HOAConsumerException("Automaton has universal branching, not supported");
    }

    std::size_t from = stateId;
    std::size_t to = conjSuccessors.front();
    createStatesUpTo(to);

    if (labelExpr) {    
      // std::cout << labelExpr->toString() << std::endl;
      EdgeCreator ec(from, to, *_nba);
      LTLFormula_ptr guard_dnf=label2dnf(labelExpr);
      guard_dnf->forEachMonom(ec);
    } else {
      throw cpphoafparser::HOAConsumerException("No label on transition!");
    }
  }

  /**
   * Called by the parser to notify the consumer that the definition for state `stateId`
   * has ended [multiple].
   */
  virtual void notifyEndOfState(unsigned int stateId) override {
    // nothing to do
  }

  /**
   * Called by the parser to notify the consumer that the automata definition has ended [mandatory, once].
   */
  virtual void notifyEnd() override {
    if (!_haveStartState || _startState >= _numberOfStates) {
      // no start state = rejecting automaton
      _startState = _nba->nba_i_newState();
      LTLNode_p node_true(new LTLNode(LTLNode::T_TRUE,
				      LTLNode_p(),
				      LTLNode_p()));
      LTLFormula_ptr guard_true(new LTLFormula(node_true, _nba->nba_i_getAPSet()));
      EdgeCreator ec(_startState, _startState, *_nba);
      guard_true->forEachMonom(ec);
    }

    _nba->nba_i_setStartState(_startState);
  }


  /**
   * Called by the parser to notify the consumer that an "ABORT" message has been encountered 
   * (at any time, indicating error, the automaton should be discarded).
   */
  virtual void notifyAbort() {
    throw cpphoafparser::HOAConsumerException("Incomplete automaton, aborted");
  }

  /**
   * Is called whenever a condition is encountered that merits a (non-fatal) warning.
   * The consumer is free to handle this situation as it wishes.
   */
  virtual void notifyWarning(const std::string& warning) {
    std::cerr << "Warning: " << warning << std::endl;
  }


private:
  NBA_I* _nba;
  APSet_p _apset;
  APSet_cp _target_apset;
  bool _isBuchi;
  bool _isAllAccepting;
  bool _isNoneAccepting;
  bool _haveStartState;
  bool _knowNumberOfStates;
  std::size_t _hoaNumberOfStates;
  std::size_t _startState;
  std::size_t _numberOfStates;
  std::vector<std::string>& _properties;
  std::vector<std::size_t> _apMapping;

  HOA2NBA(NBA_I* nba, std::vector<std::string>& properties, APSet_cp target_apset) 
    : _nba(nba),
      _apset(new APSet),
      _target_apset(target_apset),
      _isBuchi(false),
      _isAllAccepting(false),
      _isNoneAccepting(false),
      _haveStartState(false),
      _knowNumberOfStates(false),
      _hoaNumberOfStates(0),
      _startState(0),
      _numberOfStates(0),
      _properties(properties) {}

  /** Ensure that state index and all states with lesser index exist. */
  void createStatesUpTo(std::size_t index) {
    if (_knowNumberOfStates && index >= _hoaNumberOfStates) {
      throw cpphoafparser::HOAConsumerException("Automaton specified 'States: "
						+ boost::lexical_cast<std::string>(_hoaNumberOfStates)
						+ ", but references a state with index "
						+ boost::lexical_cast<std::string>(index));
    }
    while (_numberOfStates <= index) {
      std::size_t name = _nba->nba_i_newState();
      assert(name == _numberOfStates);
      if (_isAllAccepting) {
	_nba->nba_i_setFinal(name, true);
      }
      _numberOfStates++;
    }
  }

  /** Convert label expression to LTL formula in DNF */
  LTLFormula_ptr label2dnf(label_expr::ptr expr) {
    LTLFormula_ptr ltl(new LTLFormula(label2ltl(expr), _nba->nba_i_getAPSet()));
    ltl = ltl->toPNF();
    ltl = ltl->toDNF();
    return ltl;
  }

  /** Convert label expression to LTL formula */
  LTLNode_p label2ltl(label_expr::ptr expr) {
    switch (expr->getType()) {
    case label_expr::EXP_AND: {
      LTLNode_p left=label2ltl(expr->getLeft());
      LTLNode_p right=label2ltl(expr->getRight());
      return LTLNode_p(new LTLNode(LTLNode::T_AND, left, right));
    }
    case label_expr::EXP_OR: {
      LTLNode_p left=label2ltl(expr->getLeft());
      LTLNode_p right=label2ltl(expr->getRight());
      return LTLNode_p(new LTLNode(LTLNode::T_OR, left, right));
    }
    case label_expr::EXP_NOT: {
      LTLNode_p left=label2ltl(expr->getLeft());
      return LTLNode_p(new LTLNode(LTLNode::T_NOT, left));
    }
    case label_expr::EXP_TRUE:
      return LTLNode_p(new LTLNode(LTLNode::T_TRUE, 
				   LTLNode_p(),
				   LTLNode_p()));
    case label_expr::EXP_FALSE:
      return LTLNode_p(new LTLNode(LTLNode::T_FALSE, 
				   LTLNode_p(), 
				   LTLNode_p()));

    case label_expr::EXP_ATOM: {
      if (expr->getAtom().isAlias()) {
	throw cpphoafparser::HOAConsumerException("Unresolved alias in label expression");
      }
      if (_target_apset) {
	std::size_t mappedIndex = _apMapping.at(expr->getAtom().getAPIndex());
	return LTLNode_p(new LTLNode(mappedIndex));
      } else {
	return LTLNode_p(new LTLNode(expr->getAtom().getAPIndex()));
      }
    }
    default:
      throw cpphoafparser::HOAConsumerException("Unknown element in label expression");
    }
  }  

  /**
   * If there is a target APSet, generate the mapping between the AP index
   * in the HOA automaton and the target APSet
   */
  void generateAPMapping() {
    for (unsigned int i=0;i<_apset->size();i++) {
      int targetAP=_target_apset->find(_apset->getAP(i));
      if (targetAP < 0) {
	throw cpphoafparser::HOAConsumerException("Unexpected atomic proposition in HOA automaton: "+
						  _apset->getAP(i));
      }
      _apMapping.push_back((unsigned int)targetAP);
    }
  }


  /** Functor to create the edges. */
  struct EdgeCreator {
    std::size_t _from, _to;
    NBA_I& _nba;
    
    explicit EdgeCreator(std::size_t from,
			 std::size_t to,
			 NBA_I& nba) : 
      _from(from), _to(to), _nba(nba) {}
    
    void operator()(APMonom& m) {
      _nba.nba_i_addEdge(_from, m, _to);
    }
  };
    


};

#endif
