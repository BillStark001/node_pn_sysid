function net = network()


    net = power_network();
    net.omega0 = 2 * 60 * pi;

    % buses

    shunt = [0, 0];

    bus_1 = bus.slack(2, 0, shunt);
    net.add_bus(bus_1);

    bus_2 = bus.PV(0.5, 2, shunt);
    net.add_bus(bus_2);

    % branch

    branch12 = branch.pi(1, 2, [0.010, 0.085], 0);
    net.add_branch(branch12);

    % generators

    Xd = 0.963; Xd_p = 0.963; Xq = 0.963; Td_p = 5.14; M = 100; D = 10;
    mac_data = table(Xd, Xd_p, Xq, Td_p, M, D);
    component1 = component.generator.classical(mac_data);
    net.a_bus{1}.set_component(component1);

    Xd = 0.667; Xd_p = 0.667; Xq = 0.667; Td_p = 8.97; M = 12; D = 10;
    mac_data = table(Xd, Xd_p, Xq, Td_p, M, D);
    comp2 = component.generator.classical(mac_data);
    net.a_bus{2}.set_component(comp2);

end
