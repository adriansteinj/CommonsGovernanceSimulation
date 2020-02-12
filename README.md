# Commons Governance Simulation
A Python-based agent-based model of a community governing a common resource through reputation and identity.

**Common-pool resources**

Common-pool resources are publicly available, collectively-managed resources that “add [...] to our sense of belonging.” (Standing, 2019, p.32). The resources that such a system can govern can be of five types: 
- Natural commons – common natural resources like water, land or wind, now often expanded to our planetary climate system; 
- Social commons – community-owned social services like housing, transport services, and healthcare; 
- Civil commons – public institutions of justice that rely on due process and rule-following of all agents involved; 
- Cultural commons –  cultural goods that thrive through collective exchange and enjoyment like arts, sports, media and the infrastructures for them; 
- Knowledge commons – publicly exchanged information, ideas and learning processes

**Cooperation mechanisms**

There are two relevant mechanisms that can support successful community-based resource governance. 
- A reputation-based mechanism, where individual agents develop trust towards each other through conditional reciprocity, leading to cooperation or punishment for defectors. 
- An identity-based mechanism that relies on social norms, which agents learn through observation of others, and which interact with each agent's identification with and affective commitment towards the group. 

**Theoretical hypotheses**


These two parallel mechanisms together constitute a novel cognitive-behavioral framework of CPR governance which yields testable hypotheses about the effects of several key intervention variables on system-level beliefs and behaviors.

These hypotheses are:
1. The reputation and identity mechanisms in isolation will result in less cooperation and less sustainable resource use than the combination of the two mechanisms (Bergami & Bagozzi, 2000). 
2. Cooperation should be higher in groups with homogeneously high levels of identification than in groups with more heterogeneous levels of identification (Habyarimana et al., 2007).
3. Increased resource levels should correlate with increased identification and affective commitment as the system moves from an exploitation towards a conservation phase (Mosimane et al., 2012). 
4. Revealing everyone’s current strategy for resource use should increase cooperation through reputation mechanisms in groups larger than 20 (Aflagah et al., 2019).
5. Directed, incomplete networks should sustain cooperation levels equally as high as complete networks while maintaining more idealistic social norms due to overestimation of others’ appropriation (Shreedhar et al., 2018).

**System and agents**

Simulating a complex system with ecological, social, economic, and cognitive variables requires making some simplifications and assumptions, which follow from the literature. The model looks at a resource-using community with clear boundaries to its users, as Ostrom (1990) suggests. There are several micro-situational variables that are set to fixed or initial levels. These include:
- Network size, which is bounded at $ N = 20 $, to implement a community small enough to show cooperation without system-wide communication, as predicted in Aflagah et al. (2019). 
- Resource stock, which starts at level $ R = 100 $
- The rate of return $ r_i $, denoting the natural increase in the resource per community member at each time period. For sake of simplicity, the model assumes $r_i=1$ resource unit. As a result, the resource naturally increases at the rate $ R(t) = R(t-1) + N r_i $

Following the cognitive-behavioral framework of CPR governance, each agent holds a number of beliefs which produce behavior. The core variables encoding these beliefs are:
- Social norms distribution of expected cooperation $ p (c) $
- Affective commitment towards cooperation $ A $
- Identification with the community $ Q $

**Note to reader**

The current implementation does not yet implement a fully fleshed-out reputation mechanism and can thus not compare it to the social norm mechanism. Furthermore, it does not yet include a formula for personal interaction and its effects on affective commitment.
