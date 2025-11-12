library(shiny)

# Load the Philadelphia data
philly_data <- read.csv("data_philly.csv", stringsAsFactors = FALSE)

# Clean and prepare data
philly_data$Year <- as.numeric(philly_data$Year)
philly_data <- philly_data[!is.na(philly_data$Year), ]

# Get unique types and years for dropdown menus
venue_types <- c("All", sort(unique(unlist(strsplit(philly_data$type, ",")))))
venue_types <- venue_types[venue_types != "" & !is.na(venue_types)]
years <- c("All", sort(unique(philly_data$Year)))

ui <- fluidPage(
  titlePanel("Philadelphia Historical Venues - Data Explorer"),
  
  sidebarLayout(
    sidebarPanel(
      # Multiple selectInput dropdowns
      selectInput("venue_type", 
                  "Filter by Venue Type:",
                  choices = venue_types,
                  selected = "All"),
      
      selectInput("year", 
                  "Filter by Year:",
                  choices = years,
                  selected = "All"),
      
      numericInput("rows", 
                   "Number of rows to display:", 
                   value = 10, 
                   min = 1)
    ),
    
    mainPanel(
      h3("Data Preview"),
      tableOutput("table"),
      
      h3("Summary Statistics"),
      verbatimTextOutput("summary")
    )
  )
)

server <- function(input, output) {
  # Reactive expression - filters data based on multiple inputs
  # This avoids duplicate filtering code and is reused by both outputs
  filtered_data <- reactive({
    data <- philly_data
    
    # Filter by venue type
    if (input$venue_type != "All") {
      data <- data[grepl(input$venue_type, data$type, fixed = TRUE), ]
    }
    
    # Filter by year
    if (input$year != "All") {
      data <- data[data$Year == as.numeric(input$year), ]
    }
    
    return(data)
  })
  
  # renderTable() - displays data in a table format
  output$table <- renderTable({
    head(filtered_data(), n = input$rows)
  })
  
  # renderPrint() - displays text output (summary statistics)
  output$summary <- renderPrint({
    data <- filtered_data()
    cat("Total venues:", nrow(data), "\n")
    cat("Year range:", if(nrow(data) > 0) paste(min(data$Year), "-", max(data$Year)) else "N/A", "\n")
    cat("\nColumn summaries:\n")
    summary(data[, c("Year", "type", "city")])
  })
}

shinyApp(ui = ui, server = server)