% A code for event detection using STA/LTA trigger
% Time Series data imported from the CSV file as a table
% time = table values for the time
% sensval = Output values of the geophone sensor (velocity in m/s)


t=table2array(time);
val=table2array(sensval);

figure
plot(t,val)
title("Time Series Data of the geophone");
xlabel("t (in s)");
ylabel("Velocity (in m/s)");

figure
plot(t(35801:38000,1),val(35801:38000,1));
title("Time Series Data of the geophone (zoomed)");
xlabel("t (in s)");
ylabel("Velocity (in m/s)");

sta=zeros(55000,1);
lta=zeros(55000,1);
ratio=zeros(55000,1);

for i=5001:60000
    for j=1:50
        sta(i-5000)=sta(i-5000)+val(i-j);
    end
    for j=1:5000
        lta(i-5000)=lta(i-5000)+val(i-j);
    end
    sta(i-5000)=sta(i-5000)/50;
    lta(i-5000)=lta(i-5000)/5000;
    ratio(i-5000) = sta(i-5000)/lta(i-5000);
end

new_t=t(5001:60000,1);

figure
plot(new_t,sta);
title("STA with time");
xlabel("t (in s)");
ylabel("Short Term Average");

figure
plot(new_t,lta);
title("LTA with time");
xlabel("t (in s)");
ylabel("Long Term Average");

figure
plot(new_t,ratio);
title("Ratio of STA to LTA with time");
xlabel("t (in s)");
ylabel("STA/LTA");

new_t2=t(35801:38000,1);
new_ratio=ratio(30801:33000,1);

figure
plot(new_t2,new_ratio);
title("Ratio of STA to LTA with time (zoomed)");
xlabel("t (in s)");
ylabel("STA/LTA");




        