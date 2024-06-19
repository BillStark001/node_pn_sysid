function dydt = network_2bus2gen_ode(inputs, params, G)
    y = inputs.y;
    
    % TODO determine size dynamically?
    dydt = G.array(zeros(4, 1), 'dydt');
    omega0 = params.omega0;

    dydt(1) = y(2);
    dydt(3) = y(4);

    % TODO temporal solution; this will fail if the problem setting is changed

    P1 = -params.V1 * (params.V2 * ( ...
        params.B * G.sin(y(1) - y(3)) - params.G * G.cos(y(1) - y(3)) ...
    ) - params.V1 * params.G11);

    dydt(2) = (-params.D1 * y(2) / omega0 - P1 + params.Pmech1) * omega0 / params.M1;

    P2 = -params.V2 * (params.V1 * ( ...
        params.B * G.sin(y(3) - y(1)) - params.G * G.cos(y(3) - y(1)) ...
    ) - params.V2 * params.G22);

    dydt(4) = (-params.D2 * y(4) / omega0 - P2 + params.Pmech2) * omega0 / params.M2;

    return;
end
