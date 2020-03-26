# Commons Governance Simulation
A Python implementation of an agent-based model of a community governing a common resource through reputation and identity.

**Common-pool resources**

Common-pool resources are publicly available, collectively-managed resources that “add [...] to our sense of belonging.” (Standing, 2019, p.32). The resources that such a system can govern can be of five types: 
- Natural commons – common natural resources like water, land or wind, now often expanded to our planetary climate system; 
- Social commons – community-owned social services like housing, transport services, and healthcare; 
- Civil commons – public institutions of justice that rely on due process and rule-following of all agents involved; 
- Cultural commons –  cultural goods that thrive through collective exchange and enjoyment like arts, sports, media and the infrastructures for them; 
- Knowledge commons – publicly exchanged information, ideas and learning processes

**Cooperation mechanisms**

There are two relevant mechanisms that can support successful community-based resource governance. 
- A reputation-based mechanism, where individual agents develop trust towards each other through learning about the individuals around them and cooperating only conditional upon the reputation of others, leading to cooperation or punishment for defectors. 
- An identity-based mechanism that relies on social norms, which agents learn through observation of others, and which interact with each agent's identification with and affective commitment towards the group. 

The implementation of the model framework with these two cooperation mechanisms is in the [cprsim.py](cprsim.py) file.

**Theoretical hypotheses**

These two parallel mechanisms together constitute a novel cognitive-behavioral framework of CPR governance which yields testable hypotheses about the effects of several key intervention variables on system-level beliefs and behaviors.

These hypotheses are:
1. Increased resource levels should correlate with increased identification and affective commitment as the system moves from an exploitation towards a conservation phase (Mosimane et al., 2012).
2. Directed, incomplete networks should sustain cooperation levels equally as high as complete networks while maintaining more idealistic social norms due to overestimation of others’ appropriation (Shreedhar et al., 2018). 
3. The reputation and identity mechanisms in isolation will result in less cooperation and less sustainable resource use than the combination of the two mechanisms (Bergami & Bagozzi, 2000). 
4. Social interaction should increase the sense of belonging that community members feel and increase identity-based cooperation (Jussila et al., 2012).
5. Cooperation should be higher in groups with homogeneously high levels of identification than in groups with more heterogeneous levels of identification (Habyarimana et al., 2007).
6. In groups without strong social norms, revealing everyone’s current strategy for resource use should increase cooperation (Aflagah et al., 2019).

The [Testing Cooperation Hypotheses](testing_cooperation_hypotheses.ipynb) Notebook shows the testing of these hypotheses with the common-pool resource governance model.
