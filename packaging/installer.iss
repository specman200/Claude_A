; Inno Setup script for the PPE station installer.
;
;   iscc /DAppVersion=1.0.0 packaging\installer.iss      (after the PyInstaller build)
;
; Output: dist\PPEStation-Setup-<version>.exe
;
; Interactive install: asks for the machine number and fills in that
; machine's relay module address from the IP plan.
; Silent install (IT deployment), one line per PC:
;
;   PPEStation-Setup-1.0.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /MACHINE=7
;   PPEStation-Setup-1.0.0.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /RELAYIP=192.168.1.56
;
; Upgrades: run a newer installer the same way. The program is replaced; the
; machine's config.yaml, logs and captures in ProgramData are left alone
; (pass /MACHINE= or /RELAYIP= again only to change the relay address).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppName      "PPE Station"
#define AppExe       "PPEStation.exe"
#define ConsoleExe   "PPEStation-console.exe"
; IP plan: machine N's relay module is IpPrefix + (RelayBase + N - 1).
#define Machines     32
#define IpPrefix     "192.168.1."
#define RelayBase    50

[Setup]
AppId={{6F3B1C2E-8A4D-4E7B-9C1F-2D5A7E9B3C41}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=[Owner]
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=PPEStation-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Close a running station (and its supervisor) before replacing its files;
; do not restart it from the installer's own session.
CloseApplications=force
RestartApplications=no
UninstallDisplayIcon={app}\{#AppExe}
SetupLogging=yes

[Tasks]
Name: "autostart"; Description: "Start the station automatically when a user logs on"; GroupDescription: "Startup:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Dirs]
; Operators must be able to save settings from the UI and write logs.
Name: "{commonappdata}\{#AppName}"; Permissions: users-modify
Name: "{commonappdata}\{#AppName}\logs"; Permissions: users-modify

[Files]
Source: "..\dist\PPEStation\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; This machine's config: only on a fresh install, never overwritten or removed.
Source: "..\config.yaml"; DestDir: "{commonappdata}\{#AppName}"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
; --supervise everywhere: a crashed station restarts itself; a second launch
; while one is running exits at once (one station per PC).
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"; Parameters: "--supervise"; WorkingDir: "{commonappdata}\{#AppName}"
Name: "{group}\{#AppName} (debug view)"; Filename: "{app}\{#AppExe}"; Parameters: "--supervise --debug"; WorkingDir: "{commonappdata}\{#AppName}"
Name: "{group}\Self-test"; Filename: "{cmd}"; Parameters: "/k ""{app}\{#ConsoleExe}"" --selftest"; WorkingDir: "{commonappdata}\{#AppName}"
Name: "{group}\Logs folder"; Filename: "{commonappdata}\{#AppName}\logs"
Name: "{group}\Edit configuration"; Filename: "notepad.exe"; Parameters: """{commonappdata}\{#AppName}\config.yaml"""
Name: "{commonstartup}\{#AppName}"; Filename: "{app}\{#AppExe}"; Parameters: "--supervise"; WorkingDir: "{commonappdata}\{#AppName}"; Tasks: autostart
Name: "{commondesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Parameters: "--supervise"; WorkingDir: "{commonappdata}\{#AppName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Parameters: "--supervise"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{sys}\taskkill.exe"; Parameters: "/F /IM {#AppExe}"; Flags: runhidden; RunOnceId: "StopStation"

[Code]
var
  MachinePage: TInputQueryWizardPage;
  ConfigExisted: Boolean;
  AutoIP: String;   { the address last filled in from the machine number }

function ConfigPath(): String;
begin
  Result := ExpandConstant('{commonappdata}\{#AppName}\config.yaml');
end;

function RelayForMachine(N: Integer): String;
begin
  Result := '{#IpPrefix}' + IntToStr({#RelayBase} + N - 1);
end;

function IsValidIPv4(S: String): Boolean;
var
  I, Part, Dots: Integer;
  Digits: String;
begin
  Result := False;
  Dots := 0;
  Digits := '';
  S := S + '.';
  for I := 1 to Length(S) do
  begin
    if S[I] = '.' then
    begin
      if (Length(Digits) = 0) or (Length(Digits) > 3) then Exit;
      Part := StrToIntDef(Digits, -1);
      if (Part < 0) or (Part > 255) then Exit;
      Dots := Dots + 1;
      Digits := '';
    end
    else if (S[I] >= '0') and (S[I] <= '9') then
      Digits := Digits + S[I]
    else
      Exit;
  end;
  Result := Dots = 4;
end;

{ The relay address this install should write, or '' to leave the config alone. }
function WantedRelayIP(): String;
var
  Machine: Integer;
begin
  Result := Trim(ExpandConstant('{param:RELAYIP|}'));
  if Result <> '' then Exit;
  if not WizardSilent() and (MachinePage <> nil) and not ConfigExisted then
  begin
    Result := Trim(MachinePage.Values[1]);
    Exit;
  end;
  Machine := StrToIntDef(ExpandConstant('{param:MACHINE|}'), 0);
  if Machine > 0 then
    Result := RelayForMachine(Machine);
end;

function InitializeSetup(): Boolean;
begin
  ConfigExisted := FileExists(ConfigPath());
  Result := True;
end;

procedure InitializeWizard();
begin
  MachinePage := CreateInputQueryPage(wpSelectTasks,
    'Machine', 'Which machine is this PC for?',
    'Enter the machine number from the IP plan (1 to {#Machines}). The relay module address ' +
    'is filled in from it; change it only if the IP plan says otherwise.');
  MachinePage.Add('Machine number:', False);
  MachinePage.Add('Relay module IP address:', False);
  MachinePage.Values[0] := ExpandConstant('{param:MACHINE|}');
  MachinePage.Values[1] := ExpandConstant('{param:RELAYIP|}');
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  { An upgrade keeps this machine's settings; the page only appears on a first install. }
  Result := (PageID = MachinePage.ID) and ConfigExisted;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Machine: Integer;
begin
  Result := True;
  if CurPageID <> MachinePage.ID then Exit;
  Machine := StrToIntDef(Trim(MachinePage.Values[0]), 0);
  if (Machine < 1) or (Machine > {#Machines}) then
  begin
    MsgBox('Enter a machine number from 1 to {#Machines}.', mbError, MB_OK);
    Result := False;
    Exit;
  end;
  { Follow the machine number unless someone typed an address of their own. }
  if (Trim(MachinePage.Values[1]) = '') or (Trim(MachinePage.Values[1]) = AutoIP) then
  begin
    AutoIP := RelayForMachine(Machine);
    MachinePage.Values[1] := AutoIP;
  end;
  if not IsValidIPv4(Trim(MachinePage.Values[1])) then
  begin
    MsgBox('The relay module IP address is not a valid IPv4 address.', mbError, MB_OK);
    Result := False;
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine;
  if MemoTasksInfo <> '' then
    Result := Result + MemoTasksInfo + NewLine + NewLine;
  if ConfigExisted then
    Result := Result + 'Machine settings:' + NewLine + Space +
              'kept from the existing ' + ConfigPath()
  else
    Result := Result + 'Machine settings:' + NewLine +
              Space + 'Machine number: ' + Trim(MachinePage.Values[0]) + NewLine +
              Space + 'Relay module IP address: ' + Trim(MachinePage.Values[1]);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  IP, Msg: String;
  Code: Integer;
begin
  if CurStep <> ssPostInstall then Exit;

  IP := WantedRelayIP();
  if IP <> '' then
  begin
    if not IsValidIPv4(IP) then
      Msg := 'Relay address "' + IP + '" is not a valid IPv4 address; config left unchanged.'
    else if not Exec(ExpandConstant('{app}\{#ConsoleExe}'), '--set-relay-ip ' + IP,
                     ExpandConstant('{commonappdata}\{#AppName}'), SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then
      Msg := 'Could not write the relay address ' + IP + ' into ' + ConfigPath() +
             ' (exit code ' + IntToStr(Code) + '). Set tower.host by hand.'
    else
      Msg := '';
    Log('relay address ' + IP + ': ' + Msg);
    if (Msg <> '') and not WizardSilent() then
      MsgBox(Msg, mbError, MB_OK);
  end
  else if not ConfigExisted then
    Log('no machine number or relay address given; config keeps the default relay address');

  { Load the model once from the installed files: a broken install shows up now, not on the shop floor. }
  if not Exec(ExpandConstant('{app}\{#ConsoleExe}'), '--selftest',
              ExpandConstant('{commonappdata}\{#AppName}'), SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then
  begin
    Msg := 'The installed station failed its self-test (exit code ' + IntToStr(Code) + '). ' +
           'See ' + ExpandConstant('{commonappdata}\{#AppName}\logs\station.log') + '.';
    Log(Msg);
    if not WizardSilent() then
      MsgBox(Msg, mbError, MB_OK);
  end
  else
    Log('self-test passed');
end;
