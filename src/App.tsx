import React from "react";
import logo from "./logo.svg";
import "./App.css";
import { Client as KustoClient } from "azure-kusto-data";
import { KustoConnectionStringBuilder } from "azure-kusto-data";

function App() {
  const executeKustoQuery = async () => {
    const clusterName = "icmcluster";
    const kcsb = KustoConnectionStringBuilder.withUserPrompt(
      `https://${clusterName}.kusto.windows.net`,
      {
        redirectUri:
          "https://management.azure.com/subscriptions/9800ba89-8171-4da9-ba78-f42bfe20d3e7/resourceGroups/mms-sea/providers/Microsoft.Automation/automationAccounts/KustotoICMCluster",
        clientId: "94c7a459-3bfc-4408-bc10-f501fbb43e09",
      },
    );
    const client = new KustoClient(kcsb);
    const results = await client.execute("db", "Incidents | take 10");
    console.log(JSON.stringify(results));
    console.log(results.primaryResults[0].toString());
  };

  executeKustoQuery();

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
