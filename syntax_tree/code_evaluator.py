from typing import TypeVar, Generic, List

import abc

T = TypeVar("T")

class CodeEvaluator(Generic[T], abc.ABC):
  
  
  @abc.abstractmethod
  def eval_matrix_literal(self) -> T:
    pass
    
  @abc.abstractmethod
  def eval_char_literal(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_string_literal(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_cell_literal(self) -> T:
    pass
  
  
  @abc.abstractmethod
  def eval_arithmetic_opr(self, opr: str, *children: List[T]) -> T:
    pass
  
  @abc.abstractmethod
  def eval_compare_opr(self, opr: str, *children: List[T]) -> T:
    pass
    
  @abc.abstractmethod
  def eval_logical_opr(self, opr: str, *children: List[T]) -> T:
    pass
    
  
  @abc.abstractmethod
  def eval_slice(self, opr1, opr2, opr3) -> T:
    pass
  
  @abc.abstractmethod
  def eval_end_index(self) -> T:
    pass
  
  
  @abc.abstractmethod
  def eval_subs_ref_arr(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_subs_assign_arr(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_subs_ref_cell(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_subs_assign_cell(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_subs_ref_obj(self) -> T:
    pass
  
  @abc.abstractmethod
  def eval_subs_assign_obj(self) -> T:
    pass
  
  
  