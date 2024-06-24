function dydt = network_2bus2gen_ode(inputs, params, G)
    y = inputs.y;
    
    omega_0 = params.omega_0;

    % TODO temporal solution; this will fail if the problem setting is changed

    P_1 = -params.V_field_1 * (params.V_field_2 * ( ...
        params.B_12 * G.sin(y(1) - y(3)) - params.G_12 * G.cos(y(1) - y(3)) ...
    ) - params.V_field_1 * params.G_11);

    dydt_2 = (-params.D_1 * y(2) / omega_0 - P_1 + params.P_mech_1) * omega_0 / params.M_1;

    P_2 = -params.V_field_2 * (params.V_field_1 * ( ...
        params.B_12 * G.sin(y(3) - y(1)) - params.G_12 * G.cos(y(3) - y(1)) ...
    ) - params.V_field_2 * params.G_22);

    dydt_4 = (-params.D_2 * y(4) / omega_0 - P_2 + params.P_mech_2) * omega_0 / params.M_2;

    dydt = [
        y(2);
        dydt_2;
        y(4);
        dydt_4;
    ];

    return;
end
