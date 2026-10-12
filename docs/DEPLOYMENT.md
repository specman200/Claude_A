# Deploying the PPE Station to the machines

One installer, `PPEStation-Setup-<version>.exe`, installs the station on a
Windows 10/11 x64 PC. The same file serves all 32 machines; the only
per-machine setting, the relay module's IP address, is given at install time.

## What gets installed

| Where | What | Who can change it |
|---|---|---|
| `C:\Program Files\PPE Station\` | `PPEStation.exe` (the station, no console window), `PPEStation-console.exe` (same program with a console, for diagnosis), and `_internal\` with Python, the libraries, `models\`, `assets\` and the default `config.yaml` | Administrators only. Replaced on every upgrade |
| `C:\ProgramData\PPE Station\` | `config.yaml` (**this machine's** settings), `logs\` (`station.log`, `supervisor.log`, crash logs, latency and trial logs), `datasets\` (capture mode only) | Operators (so *Save to config* in the UI works). Never touched by an upgrade or uninstall |
| Start menu → PPE Station | PPE Station, PPE Station (debug view), Self-test, Logs folder, Edit configuration | |
| All-users Startup folder | PPE Station: starts at every logon (optional task, on by default) | |

Every shortcut starts the station with `--supervise`: if it crashes, it
restarts after 2 s, backing off to 60 s if it keeps failing. If an operator
closes the window, it stays closed until it is started again. A second launch
while one is running exits at once; only one station can own the cameras and
the relay board.

The motor stays safe through all of this because of how the station starts,
not because of the supervisor. It switches every relay output off when it
connects, and only a fresh press of the start button with compliant PPE
starts the motor. A crash or restart therefore always leaves the motor
stopped.

## 1. Build the installer

**On GitHub (recommended).** The *Windows installer* workflow
(`.github/workflows/windows-installer.yml`) runs on every push that touches
the program, on version tags (`v1.0.1`), or by hand from the Actions tab. It:

1. installs the pinned dependencies;
2. runs the packaging tests;
3. builds the program with PyInstaller and self-tests it;
4. compiles the installer;
5. installs it silently as machine 3 on the build machine, checks that
   machine 3's relay address was written, and self-tests the installed copy.

Download `PPEStation-installer` from the run's artifacts.

**On a Windows PC.** Install Python 3.11 and Inno Setup 6, then:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

**Versions.** Set the version in `ppe/__init__.py` (`__version__`) before
building a release; the installer, the uninstall entry and `--version` all
take it from there. Library versions are pinned in `packaging/constraints.txt`.
Change them only together with a full test run on a pilot station: a newer
ultralytics has already been seen to change how the model batches the two
cameras.

**Size.** Expect a program folder of roughly 1–1.5 GB, mostly PyTorch,
which ultralytics needs even when running the OpenVINO model. The installer is
compressed and is several hundred MB. The CI run shows the exact figures for
each build.

## 2. Prepare each station PC (Internal IT)

- **App Control for Business.** The station PCs enforce it, so the installed
  program must be allowed before it can run. Either:
  - deploy the installer through a **managed installer** (Intune or
    Configuration Manager), which allows what it installs automatically; or
  - add a policy built from one reference install:
    ```powershell
    New-CIPolicy -FilePath PPEStation.xml -ScanPath "C:\Program Files\PPE Station" `
                 -Level FilePublisher -Fallback Hash -UserPEs
    ```
    then merge, convert and deploy it as you do your other supplemental
    policies.

  Rebuild the policy for each new version: hash rules change with every
  build. Signing `PPEStation.exe`, `PPEStation-console.exe` and the
  installer with the company's code-signing certificate allows
  publisher-based rules that survive upgrades. Add a `SignTool=` line to
  `packaging/installer.iss`, or sign in the CI workflow.

  A block shows up in Event Viewer → Applications and Services →
  Microsoft → Windows → CodeIntegrity → Operational: event 3077 when
  enforced, 3076 in audit mode.
- **Auto logon** to the station's operator account. The station is a window
  on the desktop and needs a logged-on session; it does not run as a service.
- **BIOS:** power on after AC power loss; **Windows:** sleep and screen-off
  disabled; updates restricted to a maintenance window.
- **Network:** the IPC's own address from the IP plan. Proposed:
  `192.168.1.150` + (machine − 1), i.e. 150–181 for machines 1–32.
- **Cameras:** plugged in before the first start. Then run
  `PPEStation-console.exe camcheck --scan` to confirm Windows sees both.

## 3. Install on a machine

**Interactively:** run the installer as an administrator. On a first install
it asks for the **machine number** (1–32) and fills in that machine's relay
module address from the IP plan. Change the address only if the IP plan says
otherwise. The *Ready to install* page shows both before anything is written.

**Silently** (remote deployment, or a script per PC):

```bat
PPEStation-Setup-1.0.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /MACHINE=7
PPEStation-Setup-1.0.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /RELAYIP=192.168.1.56
```

Add `/LOG=C:\Temp\ppe-install.log` for an install log; it records the relay
address written and the self-test result.

Relay module address per machine (IP plan): `192.168.1.50` + (machine − 1).

| Machine | Relay | Machine | Relay | Machine | Relay | Machine | Relay |
|---|---|---|---|---|---|---|---|
| 1 | .50 | 9 | .58 | 17 | .66 | 25 | .74 |
| 2 | .51 | 10 | .59 | 18 | .67 | 26 | .75 |
| 3 | .52 | 11 | .60 | 19 | .68 | 27 | .76 |
| 4 | .53 | 12 | .61 | 20 | .69 | 28 | .77 |
| 5 | .54 | 13 | .62 | 21 | .70 | 29 | .78 |
| 6 | .55 | 14 | .63 | 22 | .71 | 30 | .79 |
| 7 | .56 | 15 | .64 | 23 | .72 | 31 | .80 |
| 8 | .57 | 16 | .65 | 24 | .73 | 32 | .81 |

The prefix, the starting address and the machine count are `#define`s at the
top of `packaging/installer.iss`.

**What the installer does:**

1. Copies the program to Program Files.
2. Creates `C:\ProgramData\PPE Station` with modify rights for users.
3. Writes the default `config.yaml` there, only if none exists.
4. Writes this machine's relay address into it by running
   `PPEStation-console.exe --set-relay-ip <address>`. This is a line edit, so
   every comment in the file survives.
5. Creates the shortcuts and the Startup entry.
6. Runs `PPEStation-console.exe --selftest`, which loads the model from the
   installed files and runs it once. A failure is reported (and logged) at
   install time, not on the shop floor.

## 4. Check the installation

1. Start menu → PPE Station → **Self-test**: expect
   `selftest OK — PPE Station <version>, model …, inference … ms`.
2. `PPEStation-console.exe camcheck --scan`: both cameras answer.
3. Start the station and run the I/O loop check and functional tests from
   the FAT/SAT protocols.

## 5. Upgrade

Run the newer installer the same way. Omit `/MACHINE` and `/RELAYIP`; give
them only to change the relay address. The installer closes a running
station, replaces the program and keeps `config.yaml`, logs and captures.
A silent upgrade from a system account cannot open the window on the
operator's desktop, so the station comes back at the next logon. Plan
upgrades with a restart or a log off and on.

Settings added by a new version use their built-in defaults until they are
added to the machine's `config.yaml`. The shipped default for comparison is
`C:\Program Files\PPE Station\_internal\config.yaml`.

## 6. Uninstall

Settings → Apps → PPE Station → Uninstall, or silently:
`"C:\Program Files\PPE Station\unins000.exe" /VERYSILENT`. The program is
removed; `C:\ProgramData\PPE Station` (config and logs) is kept and can be
deleted by hand.

## 7. Troubleshooting

| Symptom | Look at |
|---|---|
| Nothing happens when starting | App Control block (event 3077, above); `logs\supervisor.log` |
| Window opens, then closes and reopens | `logs\station.log`: the supervisor restarts it after each crash; native crashes are in `logs\station-crash.log` |
| "Already running" in the log | Another copy owns the station: only one per PC |
| Self-test fails | `logs\station.log`, which names the missing model file, DLL or module |
| No video | `PPEStation-console.exe camcheck --scan -v` |
| Relay not answering | `tower.host` in `config.yaml` against the IP plan; the module's own web page |

The supervisor restarts a station that **exits**. It cannot detect one that
stays open but stops working, such as a frozen window. The relay module's
communication watchdog (all outputs off within 1 s when polling stops) is
what keeps the motor safe in that case.
