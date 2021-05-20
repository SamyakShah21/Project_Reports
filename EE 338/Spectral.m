% A code for Spectral Domain Analysisi;
% Time Series data imported from the CSV file as a table
% time = table values for the time
% sensval = Output values of the geophone sensor (velocity in m/s)

t=table2array(time);
val=table2array(sensval);

freqsamp = 1000;

n=length(val);
xdft=fft(val);
xdft=xdft(1:n/2+1);
psdx=(1/(freqsamp*n))*abs(xdft).^2;
psdx(2:end-1)=2*psdx(2:end-1);
freq=0:freqsamp/length(val):freqsamp/2;

plot(freq,10*log10(psdx))
grid on
title('Power Spectral Density (Calculated Using FFT)')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (in dB/Hz)')

% Now windowing the time series data with a window length of 0.5 seconds

valwin=zeros(59500,500);

for i=501:60000
    for j=1:500
        valwin(i-500,j)=val(i-500+j);
    end
end

powrange = zeros(59500,1);

for i=1:59500
    powrange(i,1) = bandpower(valwin(i,:),freqsamp,[10,70]);
end

new_t=t(501:60000,1);

figure
plot(new_t,powrange);
title("Power with time (in 10 to 70 Hz frequency band)");
xlabel("t (in s)");
ylabel("Power (in units of input signal squared (m/s)^2)");

