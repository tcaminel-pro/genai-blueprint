

enum OpportunityStatus {
    Pursuit
    Active
    Completed
    Cancelled
}
// enum Industry {
//     Ressources_and_services @alias("Resources and Services")
//     PSD @alias("Public Sector and Defense")
//     FSI @alias("Financial Services and Insurance")
//     Manuf @alias("Manufacturing, MRT") @description("Manufacturing, abreviated as MRT. Include Spacial sector")
//     Telco @alias("Telecom, Media and Technology")
//     Health @alias("Health and Life Sciences")
// }

class ReviewedOpportunity {
  name string @description("Official name of the project/opportunity")
  opportunity_id string @description("Unique identifier from Salesforce or similar CRM")
  customer string @description("Client organization name")
  customer_segment string? @description("Industry vertical of the customer")
  status OpportunityStatus? @description("Current phase: Pursuit/Active/Completed/Cancelled")
  start_date string? @description("Planned/projected start date")
  end_date string? @description("Planned/projected end date")
  keywords string[]? @description("Key terms for semantic search")
  tech_stack_fingerprint string[]? @description("Technology combinations")

  statement_of_work KeyStatementOfWorkElement? @description("Business approach to win the opportunity")
  team Person[] @description("All involved personnel")  
  delivery DeliveryInfo @description("Operational delivery information")
  financials FinancialMetrics @description("Financial metrics")
  risks RiskAnalysis[] @description("Risk analysis")
  competition CompetitiveLandscape @description("Competitive landscape")
  bidding BiddingStrategy @description("Bidding strategy")
}

class KeyStatementOfWorkElement {
  objectives string[]? @description("Key business objectives of the project")
  scope string? @description("Scope of work description")
  requirements string[]? @description("Key requirements")
  success_metrics string[]? @description("Measurable success criteria")

}

class Person {
  name string? @description("Full name of involved person")
  role string? @description("Formal role in project")
  organization string? @description("Affiliated organization")
  contact_type string? @description("Internal/External/Client/Partner")
}

class DeliveryInfo {
  business_lines string[]? @description("Involved business units")
  locations string[]? @description("Geographic locations of delivery")
  partners string[]? @description("Third-party partners/subcontractors")
  technologies string[]? @description("Key technologies/platforms used")
}

class FinancialMetrics {
  tcv float? @description("Total Contract Value in EUR")
  annual_revenue float? @description("Yearly revenue projection")
  project_margin float? @description("Gross margin percentage")
  payment_terms string? @description("Payment schedule and conditions")
}

class RiskAnalysis {
  risk_description string? @description("Nature of the risk")
  impact_level string? @description("High/Medium/Low impact")
  mitigation_strategy string? @description("Planned mitigation approach")
  status string? @description("Current status of risk mitigation")
}

class CompetitiveLandscape {
  competitors string[]? @description("Identified competitors")
  competitive_position string? @description("Position vs competition")
  differentiators string[]? @description("Unique selling points vs competition")
}

class BiddingStrategy {
  strategy_type string? @description("Prime/Sub/JV bidding strategy")
  win_themes string[]? @description("Key winning strategy elements")
  pricing_strategy string? @description("Cost-led/Value-based pricing")
  challenges string[]? @description("Key bidding challenges")
}


function ExtractRainbow(rainbow_file: string) -> ReviewedOpportunity {
  client "DeepseekChat" 
  prompt #"
    Extract from this content:
    {{ rainbow_file }}

    {{ ctx.output_format }}
  "#
}




// Test the function with a sample resume. Open the VSCode playground to run this.
test rainbow_analysys {
  functions [ExtractRainbow]
  args {
    rainbow_file #"
# Solution Review Slides

## CNES TMA VENUS VIP_PEPS_ THEIA MUSCATE

### Barthélémy MARTI
**24/01/2019 9000559500**

# Objective of this Review Key Decision Points 



Solution | Introduction | Solution | Winning Solution | Way Forward
---|---|---|---|---
|  |  |  |  |  |
| Client - Opportunity |  |  |  |  |
| CNES TMA VENUS VIP, PEPS, THEIA MUSCATE |  |  |  |  |
| Pipeline Characteristics |  |  |  |  |
| Markets/Sectors | MRT Aerospace | CRM Opp. ID | 9000559500 |  |
| Division(s) | B1P S BS Toulouse (100%) |  |  |  |
| Opportunity Type | Fertilisation | Contract Type | Obligation of result |  |
| Contract Duration | 2+36 Months | TCV | 1,8 MK |  |
| Start Date | 01/04/2019 | CV Current Year | 0,3 MK |  |
| Decision Date | 08/03/2019 | Bid Cost | 36 | KK |
| Request Expected |  | Response Due | 30/01/2019 |  |
| Authorization | L5 (Business Unit) | Probability | 30 % |  |
| Reliability |  | Status | 0 |  |
| Scope Description |  |  |  |  |
| Opportunity | TMA des centres de production et de diffusion de données d'observation de la Terre et leur chaîne de traitement d'images associés : PEPS, THEIA MUSCATE et VENUS VIP comprenant une phase de prise de connaissances et une phase de maintenance corrective et évolutive regroupée en 3 lots avec options. L'ensemble est mono attributaire |  |  |  |
| Background & Requirements |  |  |  |  |
| Geographies |  |  |  |  |
| Solution, Transition | Prise de connaissance THEIA MUSCATE, prise de connaissance minimale sur VVENUS VIP, and/or Continuous sous-traitance avec ACS (imposée par le CNES) sur la partie VENUS VIP, PIXTART+SPACE4DEV pour la partie THEIA MUSCATE. Maintenance corrective et évolutive avec une équipe mutualisée |  |  |  |
| Staff Transfer |  | FTEs | Asset Transfer | MK |
| Global Sourcing |  | DTF/CSCs |  |  |
| Partner(s)/Subco(s) | ACS sur partie Venus/PIXSTART sur THEIA | Client Advisor |  |  |
| Accuracy |  | Status | 0 |  |
|  |  | Lead Division - Lead GBU/Country - Deal Sponsor | Updated on
---|---|---|---
|  |  | B&P S GBU FR BS Toulouse C|hristophe BRIZOT | 22/01/2109 |
| Involvement |  |  |  |  |
| (Global) Client Leader | Marc Ferrer | Sales Lead / Deal Maker | Aurore Dorez |
| Bid Manager | Barthélémy Marti | Solution Manager | Olivier Rondeau |
| Finance Lead | Caroline Jaulin/Encarnita Alemany | Deal Lawyer | Danièle Phankongsy |
| HR Lead | N/A | Procurement Lead | Yanide Pseaume |
| Contract Executive | Olivier Rondeau | (T&T) Project Manager | Sonia Gouel |
| Client Contacts & Relation Quality | Pierre Bourrousse - Stratégie Achat - Sponsor Green
|  |  | Gérard Lassalle-Valler Sponsor Green
|  |  | Sylvia Sylvander CP CNES Décideur - Amber |
| Commitment |  | Status | 0 |
| Win Plan |  |  |  |  |
| Atos Differentiators | Présent sur PEPs, Connaissances métiers et des écosystèmes d'observation de la Terre, MUNDI et partenariats |  |  |  |
| Competitors & their Differentiators | CAP : présent sur THEIA MUSCATE + proximité client
|  |  | CS : Venus (volume activité faible) mais réponse complaisance |
| Critical Issues & Mitigations |  | Thales : réponse de complaisance |
| Opportunity Status | Solution |  |  |  |
| Next Steps | Offer |  |  |  |
| Next Rainbow | Offer | Date | 25/01/2019 |  |
| Effectiveness |  | Status | 0 |
"#
  }
}