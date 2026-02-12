{
  "name": "8_kyc_check",
  "nodes": [
    {
      "parameters": {},
      "type": "n8n-nodes-base.manualTrigger",
      "typeVersion": 1,
      "position": [
        -656,
        -64
      ],
      "id": "9d51751e-21fd-4947-aafc-fcf85ccedd32",
      "name": "When clicking ‘Execute workflow’"
    },
    {
      "parameters": {
        "promptType": "define",
        "text": "=Read this ...\n\n {{ $json.content.parts[0].text }}",
        "options": {
          "systemMessage": "You are a KYC-basic checks agent.\n\nRead the Aadhar card and PAN card information. \n\nCheck if the person's name is the same in both the documents and that the birth date also matches.\n\nProvide output EXACTLY in the following JSON format:\n\n{\n  \"name\": \"string\",\n  \"aadhar_number\": \"string\",\n  \"pan_number\": \"string\",\n  \"birth_date\": \"string\",\n  \"gender\": \"string\",\n  \"address\": \"string\",\n  \"match_of_name\": \"boolean\",\n  \"match_of_birth_date\": \"boolean\"\n}\n\nDo not add any other formatting/tags to the output JSON. It should be a valid JSON."
        }
      },
      "type": "@n8n/n8n-nodes-langchain.agent",
      "typeVersion": 3.1,
      "position": [
        48,
        -64
      ],
      "id": "21eac7ef-6f28-4a46-9ed2-d542f0c2cd6a",
      "name": "AI Agent"
    },
    {
      "parameters": {
        "fileSelector": "/home/node/.n8n-files/Dummy-Aadhar-PAN.png",
        "options": {
          "dataPropertyName": "data"
        }
      },
      "type": "n8n-nodes-base.readWriteFile",
      "typeVersion": 1.1,
      "position": [
        -400,
        -64
      ],
      "id": "61e82834-76a1-4d6c-8b08-1338e7cc4dac",
      "name": "Read/Write Files from Disk"
    },
    {
      "parameters": {
        "modelName": "models/gemini-flash-latest",
        "options": {}
      },
      "type": "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
      "typeVersion": 1,
      "position": [
        48,
        112
      ],
      "id": "ef269379-67eb-48ea-a96a-78927067d0ce",
      "name": "Google Gemini Chat Model",
      "credentials": {
        "googlePalmApi": {
          "id": "7JwEKdpjfIHANu8p",
          "name": "Gemini-API-Key"
        }
      }
    },
    {
      "parameters": {
        "table": {
          "__rl": true,
          "value": "kyc_mismatch",
          "mode": "list",
          "cachedResultName": "kyc_mismatch"
        },
        "dataMode": "defineBelow",
        "valuesToSend": {
          "values": [
            {
              "column": "name",
              "value": "={{ JSON.parse($json.output).name }}"
            },
            {
              "column": "aadhar",
              "value": "={{ JSON.parse($json.output). aadhar_number}}"
            },
            {
              "column": "pan",
              "value": "={{ JSON.parse($json.output).pan_number }}"
            },
            {
              "column": "birth_date",
              "value": "={{ JSON.parse($json.output).birth_date }}"
            },
            {
              "column": "name_mismatch",
              "value": "={{ JSON.parse($json.output).match_of_name }}"
            },
            {
              "column": "dob_mismatch",
              "value": "={{ JSON.parse($json.output).match_of_birth_date }}"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.mySql",
      "typeVersion": 2.5,
      "position": [
        640,
        -192
      ],
      "id": "2de51ea0-e3aa-4ca3-ad2a-b9bad20027a7",
      "name": "Insert rows in a table",
      "credentials": {
        "mySql": {
          "id": "ujCiIyVZA4to7uqO",
          "name": "MySQL account"
        }
      }
    },
    {
      "parameters": {
        "operation": "append",
        "documentId": {
          "__rl": true,
          "value": "1lrB8cPJBDI2676Xq6GKTo0XYzSfJ6-U2gEPibSOe-C8",
          "mode": "list",
          "cachedResultName": "KYC-Completed",
          "cachedResultUrl": "https://docs.google.com/spreadsheets/d/1lrB8cPJBDI2676Xq6GKTo0XYzSfJ6-U2gEPibSOe-C8/edit?usp=drivesdk"
        },
        "sheetName": {
          "__rl": true,
          "value": "gid=0",
          "mode": "list",
          "cachedResultName": "Sheet1",
          "cachedResultUrl": "https://docs.google.com/spreadsheets/d/1lrB8cPJBDI2676Xq6GKTo0XYzSfJ6-U2gEPibSOe-C8/edit#gid=0"
        },
        "columns": {
          "mappingMode": "defineBelow",
          "value": {
            "Name": "={{ JSON.parse($json.output).name }}",
            "Aadhar Number": "={{ JSON.parse($json.output).aadhar_number }}",
            "PAN": "={{ JSON.parse($json.output).pan_number }}",
            "DOB": "={{ JSON.parse($json.output).birth_date }}"
          },
          "matchingColumns": [],
          "schema": [
            {
              "id": "Name",
              "displayName": "Name",
              "required": false,
              "defaultMatch": false,
              "display": true,
              "type": "string",
              "canBeUsedToMatch": true
            },
            {
              "id": "Aadhar Number",
              "displayName": "Aadhar Number",
              "required": false,
              "defaultMatch": false,
              "display": true,
              "type": "string",
              "canBeUsedToMatch": true
            },
            {
              "id": "PAN",
              "displayName": "PAN",
              "required": false,
              "defaultMatch": false,
              "display": true,
              "type": "string",
              "canBeUsedToMatch": true
            },
            {
              "id": "DOB",
              "displayName": "DOB",
              "required": false,
              "defaultMatch": false,
              "display": true,
              "type": "string",
              "canBeUsedToMatch": true
            }
          ],
          "attemptToConvertTypes": false,
          "convertFieldsToString": false
        },
        "options": {}
      },
      "type": "n8n-nodes-base.googleSheets",
      "typeVersion": 4.7,
      "position": [
        608,
        32
      ],
      "id": "82edc5db-4cad-40ce-b2ff-21c8e4d86de0",
      "name": "Append row in sheet",
      "credentials": {
        "googleSheetsOAuth2Api": {
          "id": "hISyalXGc1JynvZb",
          "name": "ekahate-google"
        }
      }
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "",
            "typeValidation": "strict",
            "version": 3
          },
          "conditions": [
            {
              "id": "c3b36dcc-a025-4718-b0f6-92b6323edf2a",
              "leftValue": "={{ JSON.parse($json.output).match_of_name }}",
              "rightValue": "false",
              "operator": {
                "type": "boolean",
                "operation": "false",
                "singleValue": true
              }
            },
            {
              "id": "4babefca-72b1-4cf2-b646-90ea90992fc9",
              "leftValue": "={{ JSON.parse($json.output).match_of_birth_date }}",
              "rightValue": "false",
              "operator": {
                "type": "boolean",
                "operation": "false",
                "singleValue": true
              }
            }
          ],
          "combinator": "or"
        },
        "options": {}
      },
      "type": "n8n-nodes-base.if",
      "typeVersion": 2.3,
      "position": [
        400,
        -64
      ],
      "id": "ac0b43f6-7ed5-4358-8982-53c200427ceb",
      "name": "KYC Problem?"
    },
    {
      "parameters": {
        "resource": "image",
        "operation": "analyze",
        "modelId": {
          "__rl": true,
          "value": "models/gemini-2.5-flash",
          "mode": "list",
          "cachedResultName": "models/gemini-2.5-flash"
        },
        "inputType": "binary",
        "options": {}
      },
      "type": "@n8n/n8n-nodes-langchain.googleGemini",
      "typeVersion": 1.1,
      "position": [
        -160,
        -64
      ],
      "id": "a7ba1e61-37dc-4365-8199-213ad6371491",
      "name": "Extract Aadhar and PAN from Image",
      "credentials": {
        "googlePalmApi": {
          "id": "7JwEKdpjfIHANu8p",
          "name": "Gemini-API-Key"
        }
      }
    }
  ],
  "pinData": {},
  "connections": {
    "When clicking ‘Execute workflow’": {
      "main": [
        [
          {
            "node": "Read/Write Files from Disk",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "AI Agent": {
      "main": [
        [
          {
            "node": "KYC Problem?",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Read/Write Files from Disk": {
      "main": [
        [
          {
            "node": "Extract Aadhar and PAN from Image",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Google Gemini Chat Model": {
      "ai_languageModel": [
        [
          {
            "node": "AI Agent",
            "type": "ai_languageModel",
            "index": 0
          }
        ]
      ]
    },
    "KYC Problem?": {
      "main": [
        [
          {
            "node": "Insert rows in a table",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Append row in sheet",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Extract Aadhar and PAN from Image": {
      "main": [
        [
          {
            "node": "AI Agent",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  },
  "active": false,
  "settings": {
    "executionOrder": "v1",
    "availableInMCP": false
  },
  "versionId": "3ee458b7-c1fd-449e-b684-c1a78614bc0c",
  "meta": {
    "templateCredsSetupCompleted": true,
    "instanceId": "553512e943ba7f04a9be8c6564bdf53e99760923ce4f54ddfd89c7039f74c3d7"
  },
  "id": "q6aWJk7jzOpeH5l1",
  "tags": []
}