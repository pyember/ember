-- <query description>
-- How many races did the championship winners win each year?
-- </query description>
-- <query>
SELECT
  dc.year,
  dc.name AS champion_name,
  COUNT(rw.name) AS race_wins
FROM
  drivers_championship dc
JOIN
  race_wins rw
ON
  dc.name = rw.name AND dc.year = EXTRACT(YEAR FROM TO_DATE(rw.date, 'DD Mon YYYY'))
WHERE
  dc.position = 1
GROUP BY
  dc.year, dc.name
ORDER BY
  dc.year;
-- </query>

-- <query description>
-- Compare the number of race wins vs championship positions for constructors in 2019
-- </query description>
-- <query>
WITH race_wins_2019 AS (
  SELECT team, COUNT(*) AS wins
  FROM race_wins
  WHERE EXTRACT(YEAR FROM TO_DATE(date, 'DD Mon YYYY')) = 2019
  GROUP BY team
),
constructors_positions_2019 AS (
  SELECT team, position
  FROM constructors_championship
  WHERE year = 2019
)
SELECT cp.team, cp.position, COALESCE(rw.wins, 0) AS wins
FROM constructors_positions_2019 cp
LEFT JOIN race_wins_2019 rw ON cp.team = rw.team
ORDER BY cp.position;
-- </query>

-- <query description>
-- Most race wins by a driver
-- </query description>
-- <query>
SELECT name, COUNT(*) AS win_count
FROM race_wins
GROUP BY name
ORDER BY win_count DESC
LIMIT 1;
-- </query>

-- <query description>
-- Which team won the most Constructors Championships?
-- </query description>
-- <query>
SELECT team, COUNT(*) AS championship_wins
FROM constructors_championship
WHERE position = 1
GROUP BY team
ORDER BY championship_wins DESC
LIMIT 1;
-- </query>

-- <query description>
-- Show me Lewis Hamilton's win percentage by season
-- </query description>
-- <query>
WITH hamilton_races AS (
  SELECT 
    year,
    COUNT(*) AS total_races
  FROM 
    race_results
  WHERE 
    name = 'Lewis Hamilton'
  GROUP BY 
    year
),
hamilton_wins AS (
  SELECT 
    EXTRACT(YEAR FROM TO_DATE(date, 'DD Mon YYYY')) AS year,
    COUNT(*) AS wins
  FROM 
    race_wins
  WHERE 
    name = 'Lewis Hamilton'
  GROUP BY 
    year
)
SELECT 
  hr.year,
  hr.total_races,
  COALESCE(hw.wins, 0) AS wins,
  ROUND((COALESCE(hw.wins, 0)::float / hr.total_races) * 100, 2) AS win_percentage
FROM 
  hamilton_races hr
LEFT JOIN 
  hamilton_wins hw ON hr.year = hw.year
ORDER BY 
  hr.year;
-- </query>

-- <query description>
-- Which drivers have won championships with multiple teams?
-- </query description>
-- <query>
WITH champion_teams AS (
  SELECT 
    name,
    team,
    COUNT(*) AS championships
  FROM 
    drivers_championship
  WHERE 
    position = 1
  GROUP BY 
    name, team
)
SELECT 
  name,
  COUNT(DISTINCT team) AS different_teams,
  STRING_AGG(team || ' (' || championships || ')', ', ') AS teams_with_championships
FROM 
  champion_teams
GROUP BY 
  name
HAVING 
  COUNT(DISTINCT team) > 1
ORDER BY 
  different_teams DESC, name;
-- </query>

-- <query description>
-- What tracks have hosted the most races?
-- </query description>
-- <query>
SELECT 
  venue,
  COUNT(DISTINCT date) AS race_count
FROM 
  race_wins
GROUP BY 
  venue
ORDER BY 
  race_count DESC
LIMIT 10;
-- </query>

-- <query description>
-- Compare Mercedes vs Ferrari performance in constructors championships
-- </query description>
-- <query>
SELECT 
  year,
  MAX(CASE WHEN team = 'Mercedes' THEN position ELSE NULL END) AS mercedes_position,
  MAX(CASE WHEN team = 'Mercedes' THEN points ELSE NULL END) AS mercedes_points,
  MAX(CASE WHEN team = 'Ferrari' THEN position ELSE NULL END) AS ferrari_position,
  MAX(CASE WHEN team = 'Ferrari' THEN points ELSE NULL END) AS ferrari_points
FROM 
  constructors_championship
WHERE 
  team IN ('Mercedes', 'Ferrari')
  AND year >= 2010
GROUP BY 
  year
ORDER BY 
  year;
-- </query>

-- <query description>
-- Show me the progression of fastest lap times at Monza
-- </query description>
-- <query>
SELECT 
  year,
  name,
  team,
  time,
  speed
FROM 
  fastest_laps
WHERE 
  venue = 'Monza'
ORDER BY 
  year;
-- </query>
