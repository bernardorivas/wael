function DSGRN_data_production_ex1()
% DSGRN_data_production_ex1.m
% Reads parameter sets from param_ex1.csv and, for each row with data_type='dsgrn',
% samples ICs per region, integrates with ode45 using boundary events
% on x = xBound and y = yBound, records trajectories at uniform times, plots,
% and writes a per-set CSV. No combined CSV is produced.

    baseDir   = fileparts(mfilename('fullpath'));
    paramPath = fullfile(baseDir, 'param_ex1.csv');
    logmsg("Starting DSGRN data production (ex1)...");
    t_global = tic;

    if ~isfile(paramPath)
        error("CSV file not found: %s", paramPath);
    end

    logmsg(sprintf("Loading parameters from '%s'", 'param_ex1.csv'));
    Tcsv = readtable(paramPath, 'TextType','string');

    if ~any(strcmpi(Tcsv.Properties.VariableNames,'data_type'))
        error("CSV must contain a 'data_type' column.");
    end
    dsgrn_idx = find(strcmpi(strtrim(string(Tcsv.data_type)),'dsgrn'));
    fprintf("Found %d DSGRN set(s) in %s.\n", numel(dsgrn_idx), 'param_ex1.csv');

    for k = 1:numel(dsgrn_idx)
        r = dsgrn_idx(k);
        t_set = tic;

    % Use the absolute CSV row index (1-based) for set_id so that the
    % first data row is set 1, second is set 2, etc., regardless of type
    set_id = r;
        fprintf("\n=== DSGRN set %d (CSV row %d) ===\n", set_id, r-1);

        % Domain and thresholds from CSV
        xMin = double(get_or_default(Tcsv, r, "xMin", 0.0));
        xMax = double(get_or_default(Tcsv, r, "xMax", 1.5));
        yMin = double(get_or_default(Tcsv, r, "yMin", 0.0));
        yMax = double(get_or_default(Tcsv, r, "yMax", 1.5));

        L  = parse_mat2x2(get_required(Tcsv, r, "L_np"),  "L_np",  r);
        U  = parse_mat2x2(get_required(Tcsv, r, "U_np"),  "U_np",  r);
        Th = parse_mat2x2(get_required(Tcsv, r, "Th_np"), "Th_np", r);

        xBound = Th(1,2);
        yBound = Th(2,1);

        % Time grid from CSV
        Tfinal = double(get_or_default(Tcsv, r, "Tfinal", 20.0));
        n_t    = int32(get_or_default(Tcsv, r, "n_timepoints", 2000));
        tspan_full = linspace(0, Tfinal, max(2, n_t));

        logmsg(sprintf("Set params: box=([x:%g,%g], [y:%g,%g]), bounds=(xBound:%g, yBound:%g), Tfinal=%g, n_timepoints=%d", ...
            xMin, xMax, yMin, yMax, xBound, yBound, Tfinal, n_t));

        % IC counts per region from CSV
        n1 = int32(get_or_default(Tcsv, r, "numPoints1", 0));
        n2 = int32(get_or_default(Tcsv, r, "numPoints2", 0));
        n3 = int32(get_or_default(Tcsv, r, "numPoints3", 0));
        n4 = int32(get_or_default(Tcsv, r, "numPoints4", 0));

        logmsg(sprintf("Sampling ICs per region: n1=%d, n2=%d, n3=%d, n4=%d", n1, n2, n3, n4));

        % Build initial conditions and region labels
        ICs = [];
        labels = [];

        if n1 > 0
            xr = xMin   + (xBound - xMin) * rand(double(n1), 1);
            yr = yMin   + (yBound - yMin) * rand(double(n1), 1);
            ICs    = [ICs; [xr, yr]];
            labels = [labels; ones(n1,1)];
        end
        if n2 > 0
            xr = xBound + (xMax   - xBound) * rand(double(n2), 1);
            yr = yMin   + (yBound - yMin)   * rand(double(n2), 1);
            ICs    = [ICs; [xr, yr]];
            labels = [labels; 2*ones(n2,1)];
        end
        if n3 > 0
            xr = xMin   + (xBound - xMin)   * rand(double(n3), 1);
            yr = yBound + (yMax   - yBound) * rand(double(n3), 1);
            ICs    = [ICs; [xr, yr]];
            labels = [labels; 3*ones(n3,1)];
        end
        if n4 > 0
            xr = xBound + (xMax   - xBound) * rand(double(n4), 1);
            yr = yBound + (yMax   - yBound) * rand(double(n4), 1);
            ICs    = [ICs; [xr, yr]];
            labels = [labels; 4*ones(n4,1)];
        end

        if isempty(ICs)
            fprintf("  No initial conditions requested in CSV. Skipping this set.\n");
            continue;
        end

        numTotal = size(ICs,1);
        fprintf("  Total trajectories: %d\n", numTotal);
        logmsg("Integrating trajectories...");

        % Optional: tolerances taken from CSV if present
        relTol = get_or_default_num(Tcsv, r, "relTol", 1e-10);
        absTol = get_or_default_num(Tcsv, r, "absTol", 1e-12);

        opts = odeset('Events', @(t,y) combinedEvents(t,y,xMin,xMax,yMin,yMax,xBound,yBound), ...
                      'RelTol', relTol, 'AbsTol', absTol);

        trajectories = cell(numTotal,1);
        allTrajData  = [];

        step = max(1, floor(numTotal/10));
        for i = 1:numTotal
            y0 = ICs(i,:).';
            reg_i = labels(i);

            t_current = 0;
            y_current = y0;
            t_full = [];
            y_full = [];

            while t_current < Tfinal
                % remaining requested output times
                seg_t = tspan_full(tspan_full >= t_current);
                if isempty(seg_t), break; end
                % ensure first time equals t_current
                if seg_t(1) > t_current
                    seg_t = [t_current, seg_t];
                end

                % integrate this segment
                [t_seg, y_seg, te, ye, ie] = ode45( ...
                    @(t,y) piecewiseODE(t,y,xBound,yBound,U,L), ...
                    seg_t, y_current, opts);

                % append without duplicating the join point
                if isempty(t_full)
                    t_full = t_seg;
                    y_full = y_seg;
                else
                    t_full = [t_full; t_seg(2:end)];
                    y_full = [y_full; y_seg(2:end,:)];
                end

                if isempty(te)
                    break  % reached Tfinal
                else
                    if any(ie == 1)
                        break  % left overall domain
                    else
                        % crossed interior boundary, restart at event point
                        t_current = te(1);
                        y_current = ye(1,:).';
                        continue
                    end
                end
            end

            trajectories{i} = [t_full, y_full];

            % record for CSV: Trajectory, Region, Time, x0, x1
            nP = numel(t_full);
            trajData = [ repmat(i, nP, 1), repmat(reg_i, nP, 1), t_full, y_full ];
            allTrajData = [allTrajData; trajData]; %#ok<AGROW>

            if mod(i, step) == 0 || i == numTotal
                done = i;
                elapsed = toc(t_set);
                rate = elapsed / max(done,1);
                eta  = rate * (numTotal - done);
                pct  = 100.0 * done / numTotal;
                logmsg(sprintf("Set %d: %d/%d (%5.1f%%) elapsed=%5.1fs ETA=%5.1fs", ...
                    set_id, done, numTotal, pct, elapsed, eta));
            end
        end

        logmsg(sprintf("Set %d: integration complete in %5.2fs", set_id, toc(t_set)));

        % Plot
        logmsg(sprintf("Set %d: plotting trajectories", set_id));
        f = figure('Visible','off','Position',[100 100 600 600]); ax = axes(f); hold(ax,'on');
        patch([xMin xBound xBound xMin], [yMin yMin yBound yBound], [0.8 0.8 1], 'FaceAlpha',0.5, 'EdgeColor','none');
        patch([xBound xMax xMax xBound], [yMin yMin yBound yBound], [0.8 1 0.8], 'FaceAlpha',0.5, 'EdgeColor','none');
        patch([xMin xBound xBound xMin], [yBound yBound yMax yMax], [1 0.8 0.8], 'FaceAlpha',0.5, 'EdgeColor','none');
        patch([xBound xMax xMax xBound], [yBound yBound yMax yMax], [1 1 0.8], 'FaceAlpha',0.5, 'EdgeColor','none');
        plot([xMin xMax], [yBound yBound], 'k', 'LineWidth', 2);
        plot([xBound xBound], [yMin yMax], 'k', 'LineWidth', 2);

        colors = [0 0 1; 0 0.5 0; 1 0 0; 0.85 0.65 0];
        for i = 1:numTotal
            traj = trajectories{i};
            plot(traj(:,2), traj(:,3), 'Color', colors(labels(i),:), 'LineWidth', 2);
            plot(ICs(i,1), ICs(i,2), 'ko', 'MarkerFaceColor','k');
        end
        xlabel('x'); ylabel('y');
        title(sprintf('Piecewise ODE Trajectories (set %d)', set_id));
        axis([xMin xMax yMin yMax]); axis equal; grid on;
        plot_path = fullfile(baseDir, sprintf('DSGRN_Set%d_ex1.png', set_id));
        exportgraphics(f, plot_path, 'Resolution', 150);
        close(f);
        fprintf("  • Saved plot -> %s\n", sprintf('DSGRN_Set%d_ex1.png', set_id));

        % Per-set CSV
        logmsg(sprintf("Set %d: writing per-set CSV", set_id));
        per_set_csv = fullfile(baseDir, sprintf('DSGRN_trajectories_ex1_set%d.csv', set_id));
    T_out = array2table(allTrajData, 'VariableNames', {'Trajectory','Region','Time','x0','x1'});
        writetable(T_out, per_set_csv);
        fprintf("  • Saved data -> %s\n", sprintf('DSGRN_trajectories_ex1_set%d.csv', set_id));

        logmsg(sprintf("Finished set %d in %5.2fs", set_id, toc(t_set)));
    end

    fprintf("\nAll DSGRN sets processed for exercise 1.\n");
    logmsg(sprintf("Total wall time: %5.2fs", toc(t_global)));
end

% ===========================
% Piecewise RHS and Events , BR: change to T
% ===========================
function dydt = piecewiseODE(~, y, xBound, yBound, U, L) 
    % Cross-coupled constants from CSV:
    % cx depends on y vs yBound: cx = U(2,1) if y<=yBound else L(2,1)
    % cy depends on x vs xBound: cy = U(1,2) if x<=xBound else L(1,2)
    x = y(1); v = y(2);
    if v <= yBound
        cx = U(2,1);
    else
        cx = L(2,1);
    end
    if x <= xBound
        cy = U(1,2);
    else
        cy = L(1,2);
    end
    dydt = [-x + cx; -v + cy];
end

function [value, isterm, dir] = combinedEvents(~, y, xMin, xMax, yMin, yMax, xBound, yBound)
    tol = 1e-9;
    % exit overall domain
    v1 = min([y(1) - xMin, xMax - y(1), y(2) - yMin, yMax - y(2)]) - tol;
    % interior boundaries
    v2 = y(1) - xBound;   % vertical switching line
    v3 = y(2) - yBound;   % horizontal switching line
    value  = [v1; v2; v3];
    isterm = [1; 1; 1];
    dir    = [-1; 0; 0];
end

% ===========================
% CSV helpers and logging
% ===========================
function v = get_or_default(T, r, name, defaultVal)
    if any(strcmpi(T.Properties.VariableNames, name)) && ~ismissing(T{r, name}) && ~isempty(T{r, name})
        v = T{r, name};
        if iscell(v), v = v{1}; end
    else
        v = defaultVal;
    end
end

function v = get_required(T, r, name)
    if ~any(strcmpi(T.Properties.VariableNames, name))
        error("CSV missing required column: %s", name);
    end
    v = T{r, name};
    if iscell(v), v = v{1}; end
    if ismissing(v) || isempty(v)
        error("CSV column %s is empty at row %d", name, r);
    end
end

function A = parse_mat2x2(cellstr, name, row_idx)
    s = string(cellstr);
    str = strtrim(s);
    % try JSON-like first
    try
        Atry = jsondecode(str);
        A = double(Atry);
    catch
        % fallback to MATLAB matrix syntax
        tmp = strrep(str, '],[', ';');
        tmp = strrep(tmp, '], [', ';');
        tmp = strrep(tmp, ']', '');
        tmp = strrep(tmp, '[', '');
        A = str2num(tmp); %#ok<ST2NM>
    end
    if ~isequal(size(A), [2 2])
        error("%s must be 2x2 for row %d", name, row_idx);
    end
end

function v = get_or_default_num(T, r, name, defaultVal)
    if any(strcmpi(T.Properties.VariableNames, name))
        vv = T{r, name};
        if iscell(vv), vv = vv{1}; end
        if ~ismissing(vv) && ~isempty(vv) && isfinite(double(vv))
            v = double(vv);
            return
        end
    end
    v = defaultVal;
end

function logmsg(s)
    fprintf("[%s] %s\n", datestr(now, 'HH:MM:SS'), s);
end
