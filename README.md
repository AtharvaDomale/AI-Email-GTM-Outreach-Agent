# **AI Email GTM Outreach Agent**

This is an end-to-end, multi-agent Streamlit app that automates B2B outreach using GPT-5 and Exa. It helps you discover relevant companies, find the right contacts, gather insights, and generate personalized outreach emails—all tailored to your targeting criteria.

## **Key Features**

### **Multi-Agent Workflow**:
1. **Company Finder**: 
   - Uses Exa to identify companies matching your targeting criteria and offering.
2. **Contact Finder**: 
   - Identifies 2–3 decision-makers per company (Founder’s Office, GTM/Sales leadership, Partnerships/BD, Product Marketing).
   - Provides email addresses (inferred emails are clearly marked).
3. **Researcher**: 
   - Gathers 2–4 key insights per company from their website and Reddit discussions to aid in personalized email creation.
4. **Email Writer**: 
   - Uses GPT-5 to generate concise, structured outreach emails in your chosen style.

### **Operator Controls**:
- **Target Companies**: Choose how many companies to target (1–10).
- **Email Style**: Select from four styles: 
  - Professional
  - Casual
  - Cold
  - Consultative
- **Real-Time Progress**: View the live status of the app’s workflow (Companies → Contacts → Research → Emails).

### **Security-First**:
- API keys are entered in the Streamlit sidebar and are never hardcoded or committed.

## **System Requirements**

Before running the app, install the necessary dependencies by running the following:

```bash
pip install -r requirements.txt
```

### **Environment Variables**:
- `OPENAI_API_KEY` (required for GPT-5 model access)
- `EXA_API_KEY` (required for web discovery via Exa)

These keys can be set either in the left sidebar or directly via your shell environment.

## **How to Run the App**

1. **Set API Keys**: Enter your `OPENAI_API_KEY` and `EXA_API_KEY` in the sidebar.
2. **Provide Targeting Info**: Enter a description of your target audience and offering.
3. **Choose Settings**:
   - Select how many companies to target (1–10).
   - Pick your preferred email style.
4. **Start Outreach**: Click “Start Outreach” to initiate the process.
   - The app will proceed through the stages: Companies → Contacts → Research → Emails.
5. **Review Results**: Once completed, you can:
   - View discovered companies, contacts, and research insights.
   - Download or copy the generated emails.

## **Notes**:
- The app uses GPT-5 via OpenAI. If you don’t have access to GPT-5, modify the model in the `GTM_Outreach_Agent.py` file to one you have access to.
- Exa is used for discovering companies and contacts—make sure your `EXA_API_KEY` is valid.
  
## **Troubleshooting**

- **Stalling on a Stage**: Ensure your API keys are valid and check your network connectivity.
- **JSON Parsing Errors**: If errors occur, try rerunning the stage as models can occasionally add extra text around the JSON output.
- **Rate Limits**: If you hit rate limits, reduce the number of companies you're targeting.
