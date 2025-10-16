:- module('spec', [trace_expression/2, match/2]).
:- use_module(monitor('deep_subdict')).
match(_event, a_match(N)) :- deep_subdict(_{'a':N}, _event), >(N, 0).
match(_event, b_match) :- deep_subdict(_{'b':T}, _event), T=1.0.
match(_event, c_match) :- deep_subdict(_{'c':T}, _event), T=1.0.
match(_event, d_match) :- deep_subdict(_{'d':T}, _event), T=1.0.
match(_event, not_abcd) :- not(match(_event, a_match(N))), not(match(_event, b_match)), not(match(_event, c_match)), not(match(_event, d_match)).
match(_, any).
trace_expression('Main', Main) :- Main=(star((not_abcd:eps))*var(n, ((a_match(var(n)):eps)*app(B, [var('n')])))), B=gen(['n'], (star((not_abcd:eps))*((b_match:eps)*app(C, [var('n')])))), C=gen(['n'], (star((not_abcd:eps))*((c_match:eps)*app(D, [var('n')])))), D=gen(['n'], guarded((var('n')>0), (star((not_abcd:eps))*((d_match:eps)*app(D, [(var('n')-1)]))), 1)).
