#!/usr/bin/env python3
"""Compile the production family speculative-entry guard with controlled callbacks."""
import json,subprocess,tempfile
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
s=(repo/'ds4.c').read_text();a=s.index('static int family_cont_spec(');a=s.index('    if ((!ir',a) if 'const int current_excluded' not in s[a:s.index('const uint32_t pos',a)] else s.index('    const int current_excluded',a)
b=s.index('    const uint32_t pos',a);guard=s[a:b]
code=r'''
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
struct G {int mtp_enabled;struct {void *ws;} draft;};
struct R {void *spec;struct G graph[1];};
struct B {void *step_accept;int (*sample_exclude)(void*,void*);float temperature;void *user;};
static int excluded=-1;
static int exclusion(void *a,void *b){(void)a;(void)b;return excluded;}
static struct R target,other,step;
static struct B bank;
static int iquest=1;
static int ready(void) {
 struct R *ir=iquest?&target:NULL,*nr=iquest?NULL:&other,*rt=NULL;struct B *cb=&bank;unsigned bank=0;void *ud=NULL;
 GUARD
 return 1;
}
int main(void) {
 target.graph[0].mtp_enabled=1;bank.step_accept=&bank;bank.sample_exclude=exclusion;
 int inactive=ready();excluded=42;int active=ready();excluded=-1;bank.sample_exclude=NULL;int absent=ready();
 bank.sample_exclude=exclusion;bank.temperature=0.5f;int sampling=ready();bank.temperature=0;
 setenv("DS4_MTP_SPEC_DISABLE","1",1);int disabled=ready();unsetenv("DS4_MTP_SPEC_DISABLE");
 bank.step_accept=NULL;int no_accept=ready();bank.step_accept=&bank;target.graph[0].mtp_enabled=0;int no_weights=ready();
 other.graph[0].draft.ws=&other;iquest=0;int other_unchanged=ready();
 printf("{\"inactive_callback_eligible\":%s,\"active_exclusion_blocked\":%s,\"absent_callback_eligible\":%s,\"non_greedy_blocked\":%s,\"disable_blocked\":%s,\"missing_accept_blocked\":%s,\"missing_mtp_blocked\":%s,\"other_family_callback_gate_unchanged\":%s}\n",inactive?"true":"false",!active?"true":"false",absent?"true":"false",!sampling?"true":"false",!disabled?"true":"false",!no_accept?"true":"false",!no_weights?"true":"false",!other_unchanged?"true":"false");
 return inactive && !active && absent && !sampling && !disabled && !no_accept && !no_weights && !other_unchanged?0:1;
}
'''.replace('GUARD',guard)
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'gate.c').write_text(code);subprocess.run(['cc','-std=gnu11','-O2',str(p/'gate.c'),'-o',str(p/'gate')],check=True)
 raise SystemExit(subprocess.run([str(p/'gate')]).returncode)
