
function test_creation_of_policy_given_utilities()
	# If we create a ValueIterationPolicy by providing a utility,
	# does that utility successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	# Create a starter set of utilities 
	# 0.0 is associated with the fake absorbing state
	correct_utility = [5.45632, 8.20505, 10.0, 0.0] 

	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, utility=correct_utility, include_Q=true)
	return policy.mdp == mdp && policy.util == correct_utility
end


function test_creation_of_policy_given_q_util_policy()
	# If we create a ValueIterationPolicy by providing a utility, q, and policy,
	# does those values successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	correct_utility = [5.45632, 8.20505, 10.0, 0.0] 
	correct_qmat = [5.45636 5.1835 5.1835 5.1835; 8.20506 6.47848 7.7948 7.7948; 10.0 10.0 10.0 10.0; 0.0 0.0 0.0 0.0]
	correct_policy = [1,1,1,1]
	
	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, correct_qmat, correct_utility, correct_policy)
	return (policy.mdp == mdp && policy.qmat == correct_qmat && policy.util == correct_utility && policy.policy == correct_policy && policy.include_Q == true)
end

function test_creation_of_policy_given_q()
	# If we create a ValueIterationPolicy by providing a default q,
	# does its value successfully get saved?
	#
	# GridWorld:
	# |_________|_________|_________| plus a fake state for absorbing
	mdp = GridWorld(sx=1, sy=3, rs = [GridWorldState(1,3)], rv = [10.0])
	correct_qmat = [5.45636 5.1835 5.1835 5.1835; 8.20506 6.47848 7.7948 7.7948; 10.0 10.0 10.0 10.0; 0.0 0.0 0.0 0.0]
	
	# Derive a policy & check that they match	
	policy = ValueIterationPolicy(mdp, correct_qmat)
	return (policy.mdp == mdp && policy.qmat == correct_qmat)
end

@test test_creation_of_policy_given_utilities() == true
@test test_creation_of_policy_given_q_util_policy() == true
@test test_creation_of_policy_given_q() == true