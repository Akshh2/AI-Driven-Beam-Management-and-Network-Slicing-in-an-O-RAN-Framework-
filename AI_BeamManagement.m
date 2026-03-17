%% ===============================================================
% AI-DRIVEN BEAM MANAGEMENT AND NETWORK SLICING IN AN O-RAN FRAMEWORK
% Includes:
% - Baseline scheduler
% - AI PRB allocation (Q-learning)
% - Massive MIMO
% - Beamforming
% - DQN-based beam selection
% ================================================================

clc; clear; close all;

%% ================= PARAMETERS =================
p.totalPRBs = 100;
p.nSlices = 3;
p.simTime = 200;
p.deltaT = 0.1;
p.numUsers = 60;
p.cellRadius = 500;

% Wireless parameters
p.freq = 3.5e9;
p.pathlossExp = 3;
noise = 1e-9;

% Massive MIMO parameters
p.Nt = 64;
p.Nr = 2;
p.streams = min(p.Nt, p.Nr);
p.txPower = 40;

% Beamforming parameters
p.numBeams = 8;
p.beamAngles = linspace(0,2*pi,p.numBeams+1);
p.beamAngles(end) = [];

% DQN parameters
p.stateSize = 3;
p.hiddenSize = 16;
p.learningRate = 0.01;

sliceNames = {'eMBB','URLLC','mMTC'};

%% SLA targets
target_thr = [8e6 3e6 1e6];
target_lat = [5 1 10];

%% ================= USER DISTRIBUTION =================
theta = 2*pi*rand(p.numUsers,1);
radius = p.cellRadius*sqrt(rand(p.numUsers,1));

userPos(:,1)=radius.*cos(theta);
userPos(:,2)=radius.*sin(theta);

slice_ids=[ones(1,25) 2*ones(1,15) 3*ones(1,20)];

%% ================= VISUAL SMART CITY MAP =================
figure
gscatter(userPos(:,1),userPos(:,2),slice_ids)
hold on
viscircles([0 0],p.cellRadius,'LineStyle','--');
title('Smart City User Distribution')
legend('eMBB','URLLC','mMTC')
axis equal
grid on

%% Initialize DQN (one per slice)
for s = 1:p.nSlices
    dqn{s}.W1 = randn(p.hiddenSize, p.stateSize) * 0.1;
    dqn{s}.b1 = zeros(p.hiddenSize,1);
    dqn{s}.W2 = randn(p.numBeams, p.hiddenSize) * 0.1;
    dqn{s}.b2 = zeros(p.numBeams,1);
end

%% ================= RUN SIMULATION =================
[thr_base,lat_base,prb_base,sinr_base] = ...
    simulate_baseline(p,userPos,slice_ids,noise);

[thr_ai,lat_ai,prb_ai,sinr_ai,reward_hist] = ...
    simulate_AI(p,userPos,slice_ids,target_thr,target_lat,dqn);

time = (0:p.simTime-1)*p.deltaT;

%% ================= PLOTTING =================
figure('Position',[100 100 1300 800]);

subplot(2,2,1); hold on
plot(time,thr_base/1e6,'--','LineWidth',1.5)
plot(time,thr_ai/1e6,'LineWidth',2)
title('Throughput (Mbps)')
legend('eMBB NoAI','URLLC NoAI','mMTC NoAI','eMBB AI','URLLC AI','mMTC AI')
grid on

subplot(2,2,2); hold on
plot(time,lat_base,'--','LineWidth',1.5)
plot(time,lat_ai,'LineWidth',2)
title('Latency (s)')
legend('eMBB NoAI','URLLC NoAI','mMTC NoAI','eMBB AI','URLLC AI','mMTC AI')
grid on

subplot(2,2,3); hold on
plot(time,10*log(sinr_base),'--','LineWidth',1.5)
plot(time,10*log(sinr_ai),'LineWidth',2)
title('SINR (dB)')
legend('eMBB NoAI','URLLC NoAI','mMTC NoAI','eMBB AI','URLLC AI','mMTC AI')
grid on

subplot(2,2,4); hold on
plot(time,prb_base,'--','LineWidth',1.5)
plot(time,prb_ai,'LineWidth',2)
title('PRB Allocation')
legend('eMBB NoAI','URLLC NoAI','mMTC NoAI','eMBB AI','URLLC AI','mMTC AI')
grid on

%% ================= AI CONVERGENCE =================
figure
plot(reward_hist,'LineWidth',2)
title('AI Convergence')
xlabel('Time Step')
ylabel('Reward')
grid on

%% ================= FAIRNESS INDEX =================
fair_base = (sum(mean(thr_base)).^2)/(p.nSlices*sum(mean(thr_base).^2));
fair_ai   = (sum(mean(thr_ai)).^2)/(p.nSlices*sum(mean(thr_ai).^2));

fprintf('\n===== FAIRNESS INDEX =====\n');
fprintf('Baseline: %.3f\n',fair_base);
fprintf('AI: %.3f\n',fair_ai);

%% ================= AVG COMPARISON =================
disp('===== SMART CITY AI vs BASELINE =====')

table(mean(thr_base)'/1e6,mean(thr_ai)'/1e6,...
mean(lat_base)',mean(lat_ai)',...
mean(prb_base)',mean(prb_ai)',...
'VariableNames',{'Thr_NoAI','Thr_AI','Lat_NoAI','Lat_AI','PRB_NoAI','PRB_AI'},...
'RowNames',sliceNames)

%% ===============================================================
%% BASELINE FUNCTION
%% ===============================================================
function [thr,lat,prb,sinr]=simulate_baseline(p,userPos,slice_ids,noise)

thr=zeros(p.simTime,p.nSlices);
lat=zeros(p.simTime,p.nSlices);
prb=zeros(p.simTime,p.nSlices);
sinr=zeros(p.simTime,p.nSlices);

for t=1:p.simTime

prb_slice=floor(p.totalPRBs/p.nSlices)*ones(1,p.nSlices);
traffic=[1+0.5*sin(0.05*t) 1+0.3*cos(0.03*t) 0.5+0.2*sin(0.02*t)];

for s=1:p.nSlices

users=find(slice_ids==s);
dist=sqrt(userPos(users,1).^2+userPos(users,2).^2);

PL=20*log10(p.freq)+10*p.pathlossExp*log10(dist+1);

shadowing = 10^(randn*8/10);
fading = abs(randn + 1i*randn)^2;

channel_gain = mean(fading * shadowing ./ (10.^(PL/10)));

signal = p.txPower * channel_gain * p.Nt;
interference = 0.3 * p.txPower * rand;

sinr_lin = signal / (noise + interference);
sinr(t,s)=sinr_lin;

thr(t,s)=traffic(s)*prb_slice(s)*1e5 * ...
         p.streams * log2(1+sinr_lin);

lat(t,s)=1 + 1/(thr(t,s)/1e6+0.1);

end

prb(t,:)=prb_slice;

end
end

%% ===============================================================
%% AI FUNCTION WITH DQN BEAM SELECTION
%% ===============================================================
%% ===============================================================
%% AI FUNCTION WITH Q-LEARNING + DQN BEAM SELECTION
%% ===============================================================
function [thr,lat,prb,sinr,reward_hist] = ...
simulate_AI(p,userPos,slice_ids,target_thr,target_lat,dqn)

thr=zeros(p.simTime,p.nSlices);
lat=zeros(p.simTime,p.nSlices);
prb=zeros(p.simTime,p.nSlices);
sinr=zeros(p.simTime,p.nSlices);
reward_hist=zeros(p.simTime,1);

% Q-learning parameters
alpha=0.3;
gamma=0.9;
epsilon=0.2;

numStates=5;
numActions=7;
Q=zeros(numStates,numActions);

% Possible PRB allocation actions
actions = [
    40 40 20;
    50 30 20;
    30 50 20;
    60 25 15;
    25 60 15;
    45 35 20;
    33 33 34
];

for t=1:p.simTime

traffic=[1+0.5*sin(0.05*t) ...
         1+0.3*cos(0.03*t) ...
         0.5+0.2*sin(0.02*t)];

trafficLevel = sum(traffic);
state = min(max(ceil(trafficLevel),1),5);

% Q-learning action selection
if rand<epsilon
    action=randi(numActions);
else
    [~,action]=max(Q(state,:));
end

prb_alloc = actions(action,:);

for s=1:p.nSlices

users=find(slice_ids==s);
x=userPos(users,1);
y=userPos(users,2);
dist=sqrt(x.^2+y.^2);

userAngles = atan2(y,x);
avgAngle = mean(userAngles);
avgDist = mean(dist);

stateVec = [traffic(s); avgAngle; avgDist];

% DQN forward
Qvals = dqn_forward(dqn{s}, stateVec);

% Beam selection
if rand < epsilon
    beam = randi(p.numBeams);
else
    [~, beam] = max(Qvals);
end

beamAngle = p.beamAngles(beam);
angleError = abs(wrapToPi(avgAngle - beamAngle));
alignmentGain = max(cos(angleError),0.1);

PL=20*log10(p.freq)+10*p.pathlossExp*log10(dist+1);
channel_gain = mean(1./(10.^(PL/10)));

sinr_lin = (p.txPower * channel_gain * p.Nt * alignmentGain) / 1e-9;
sinr(t,s) = sinr_lin;

thr(t,s)=traffic(s)*prb_alloc(s)*1e5 * ...
         p.streams * log2(1+sinr_lin);

lat(t,s)=1 + 1/(thr(t,s)/1e6+0.1);

% DQN training
beamReward = thr(t,s)/target_thr(s);
targetQ = Qvals;
targetQ(beam) = beamReward;

dqn{s} = dqn_train(dqn{s}, stateVec, targetQ, p.learningRate);

end

% Global reward for Q-learning
thr_score = sum(min(thr(t,:)./target_thr,1));
lat_score = sum(min(target_lat./lat(t,:),1));
reward = thr_score + lat_score;
reward_hist(t) = reward;

% Q-learning update
[~,maxQ]=max(Q(state,:));
Q(state,action)=Q(state,action)+...
alpha*(reward+gamma*Q(state,maxQ)-Q(state,action));

prb(t,:)=prb_alloc;

end
end

%% ===============================================================
%% DQN FUNCTIONS
%% ===============================================================
function Q = dqn_forward(net, state)
z1 = net.W1 * state + net.b1;
a1 = tanh(z1);
z2 = net.W2 * a1 + net.b2;
Q = z2;
end

function net = dqn_train(net, state, targetQ, lr)

z1 = net.W1 * state + net.b1;
a1 = tanh(z1);
z2 = net.W2 * a1 + net.b2;
Q = z2;

dQ = Q - targetQ;

dW2 = dQ * a1';
db2 = dQ;

da1 = net.W2' * dQ;
dz1 = da1 .* (1 - tanh(z1).^2);

dW1 = dz1 * state';
db1 = dz1;

net.W1 = net.W1 - lr * dW1;
net.b1 = net.b1 - lr * db1;
net.W2 = net.W2 - lr * dW2;
net.b2 = net.b2 - lr * db2;

end


%% ================= SITE VIEWER =================

%% ================= BASE STATION LOCATIONS =================
lat = [12.9716 12.9725 12.9705 12.9695 12.9735 12.9685];
lon = [77.5946 77.5960 77.5980 77.5930 77.5915 77.5975];
numBS = length(lat);

%% ================= NON-AI BASE STATIONS =================
for i = 1:numBS
    bs_noAI(i) = txsite( ...
        "Name","BS_noAI_"+i, ...
        "Latitude",lat(i), ...
        "Longitude",lon(i), ...
        "AntennaHeight",30, ...
        "TransmitterFrequency",3.5e9, ...
        "TransmitterPower",40);
end

%% ================= AI BASE STATIONS =================
beamAngles = [30 60 120 210 300 350];

for i = 1:numBS
    bs_AI(i) = txsite( ...
        "Name","BS_AI_"+i, ...
        "Latitude",lat(i), ...
        "Longitude",lon(i), ...
        "AntennaHeight",30, ...
        "TransmitterFrequency",3.5e9, ...
        "TransmitterPower",40);
    
    % Directional beamforming antenna
    bs_AI(i).Antenna = phased.URA([8 8]);
    bs_AI(i).AntennaAngle = [beamAngles(i) 0];
end

%% ================= PROPAGATION MODEL =================
pm = propagationModel("longley-rice");

%% ================= VIEWER 1: NON-AI =================
viewer1 = siteviewer("Basemap","satellite");
show(bs_noAI);

coverage(bs_noAI, ...
    "PropagationModel",pm, ...
    "SignalStrengths",-100:-5, ...
    "MaxRange",800, ...
    "Resolution",5);

title("Non-AI Coverage");

%% ================= VIEWER 2: AI =================
viewer2 = siteviewer("Basemap","satellite");
show(bs_AI);

coverage(bs_AI, ...
    "PropagationModel",pm, ...
    "SignalStrengths",-100:-5, ...
    "MaxRange",800, ...
    "Resolution",5);

title("AI Beamformed Coverage");